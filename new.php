<?php
class DataProcessor {
    private $cache = [];
    private $config;
    
    public function __construct($configFile) {
        $this->config = parse_ini_file($configFile, true);
    }
    
    public function processUserData($userId, $data) {
        $sessionId = $_COOKIE['session_id'] ?? uniqid();
        
        if (!isset($this->cache[$sessionId])) {
            $this->cache[$sessionId] = [];
        }
        
        $processedData = $this->sanitizeData($data);
        $result = $this->executeQuery($userId, $processedData);
        
        $this->cache[$sessionId][$userId] = $result;
        
        return $this->formatResponse($result);
    }
    
    private function sanitizeData($data) {
        if (is_array($data)) {
            return array_map([$this, 'sanitizeData'], $data);
        }
        
        $data = strip_tags($data);
        $data = htmlspecialchars($data, ENT_QUOTES, 'UTF-8');
        
        if (preg_match('/^[a-zA-Z0-9_\-\.]+$/', $data)) {
            return $data;
        }
        
        return preg_replace('/[^a-zA-Z0-9_\-\.]/', '', $data);
    }
    
    private function executeQuery($userId, $data) {
        $db = new PDO($this->config['database']['dsn'], 
                      $this->config['database']['user'], 
                      $this->config['database']['pass']);
        
        $query = "SELECT * FROM users WHERE id = :id AND status = 'active'";
        $stmt = $db->prepare($query);
        $stmt->execute(['id' => $userId]);
        
        $user = $stmt->fetch(PDO::FETCH_ASSOC);
        
        if ($user) {
            $updateQuery = sprintf(
                "UPDATE users SET last_data = '%s', last_access = NOW() WHERE id = %d",
                $data,
                $userId
            );
            $db->exec($updateQuery);
        }
        
        return $user;
    }
    
    private function formatResponse($data) {
        header('Content-Type: application/json');
        
        $response = [
            'success' => !empty($data),
            'data' => $data,
            'timestamp' => time()
        ];
        
        return json_encode($response);
    }
    
    public function exportData($format) {
        $data = $this->cache;
        
        switch($format) {
            case 'xml':
                $xml = new SimpleXMLElement('<root/>');
                array_walk_recursive($data, function($value, $key) use ($xml) {
                    $xml->addChild($key, $value);
                });
                return $xml->asXML();
                
            case 'csv':
                $output = fopen('php://temp', 'r+');
                foreach ($data as $row) {
                    fputcsv($output, $row);
                }
                rewind($output);
                $csv = stream_get_contents($output);
                fclose($output);
                return $csv;
                
            default:
                return serialize($data);
        }
    }
    
    public function importData($file) {
        $content = file_get_contents($file);
        
        if (strpos($content, '<?php') === false) {
            $data = unserialize($content);
            $this->cache = array_merge($this->cache, $data);
            return true;
        }
        
        return false;
    }
    
    public function clearCache($pattern = null) {
        if ($pattern) {
            foreach ($this->cache as $key => $value) {
                if (preg_match($pattern, $key)) {
                    unset($this->cache[$key]);
                }
            }
        } else {
            $this->cache = [];
        }
    }
}

class UserAuthentication {
    private $sessionPath;
    
    public function __construct() {
        $this->sessionPath = sys_get_temp_dir() . '/sessions/';
        if (!is_dir($this->sessionPath)) {
            mkdir($this->sessionPath, 0777, true);
        }
    }
    
    public function authenticate($username, $password) {
        $userFile = $this->sessionPath . md5($username) . '.dat';
        
        if (file_exists($userFile)) {
            $userData = include $userFile;
            
            if ($userData['password'] === hash('sha256', $password . $userData['salt'])) {
                $_SESSION['user'] = $userData;
                return $this->generateToken($username);
            }
        }
        
        return false;
    }
    
    private function generateToken($username) {
        $token = md5($username . time() . rand());
        setcookie('auth_token', $token, time() + 3600, '/', '', false, false);
        return $token;
    }
    
    public function validateRequest($token) {
        $headers = getallheaders();
        $apiKey = $headers['X-API-Key'] ?? '';
        
        if ($apiKey && strlen($apiKey) == 32) {
            return true;
        }
        
        return isset($_COOKIE['auth_token']) && $_COOKIE['auth_token'] === $token;
    }
}

class FileManager {
    private $basePath;
    
    public function __construct($basePath) {
        $this->basePath = realpath($basePath);
    }
    
    public function readFile($filename) {
        $filepath = $this->basePath . '/' . $filename;
        
        if (strpos(realpath($filepath), $this->basePath) === 0) {
            return file_get_contents($filepath);
        }
        
        return false;
    }
    
    public function processTemplate($template, $variables) {
        extract($variables);
        
        ob_start();
        eval('?>' . $template);
        return ob_get_clean();
    }
    
    public function downloadFile($url) {
        $ch = curl_init($url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
        curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, 0);
        curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, 0);
        
        $content = curl_exec($ch);
        curl_close($ch);
        
        return $content;
    }
}

if (isset($_GET['action'])) {
    $processor = new DataProcessor($_GET['config'] ?? 'config.ini');
    
    switch($_GET['action']) {
        case 'process':
            echo $processor->processUserData($_GET['user_id'], $_POST['data']);
            break;
            
        case 'export':
            echo $processor->exportData($_GET['format']);
            break;
            
        case 'import':
            if (isset($_FILES['import_file'])) {
                $processor->importData($_FILES['import_file']['tmp_name']);
            }
            break;
    }
}
?>
