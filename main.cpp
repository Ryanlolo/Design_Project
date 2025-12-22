#include <Arduino.h>

// ============================================
// Configuration - 请根据你的硬件修改这些参数
// ============================================

// ESP32与调试板通信的UART配置
#define DEBUG_BOARD_SERIAL_RX 16  // ESP32接收引脚（连接到调试板TX）
#define DEBUG_BOARD_SERIAL_TX 17  // ESP32发送引脚（连接到调试板RX）
#define DEBUG_BOARD_BAUDRATE 115200  // 与调试板通信的波特率（已确认）

// Python与ESP32通信的串口配置（USB串口）
#define USB_SERIAL_BAUDRATE 128000  // USB串口波特率

// 调试板通信选项
#define ADD_NEWLINE_AFTER_COMMAND false  // 设置为true如果调试板需要换行符
#define USE_CRLF true  // 如果ADD_NEWLINE_AFTER_COMMAND为true，使用\r\n还是\n

// 调试板通信选项
#define ADD_NEWLINE_AFTER_COMMAND false  // 设置为true如果调试板需要换行符
#define USE_CRLF true  // 如果ADD_NEWLINE_AFTER_COMMAND为true，使用\r\n还是\n

// ============================================
// 全局变量
// ============================================

HardwareSerial DebugBoardSerial(1);  // 使用UART1与调试板通信

// 指令解析缓冲区
String commandBuffer = "";

// ============================================
// 函数声明
// ============================================

/**
 * 解析单个舵机指令
 * 格式: #000P1500T1000!
 * 返回: true if valid, false otherwise
 */
bool parseServoCommand(String cmd, int& servoId, int& pwm, int& time);

/**
 * 解析多个舵机指令（用{}包裹）
 * 格式: {#000P1500T1000!#001P2500T0000!}
 */
bool parseMultiServoCommand(String cmd);

/**
 * 发送指令到调试板
 * 这里直接转发原始指令，如果调试板需要不同格式，需要在这里转换
 */
void sendToDebugBoard(String command);

/**
 * 处理接收到的完整指令
 */
void processCommand(String command);

// ============================================
// Setup函数
// ============================================

void setup() {
  // 初始化USB串口（用于与Python通信）
  Serial.begin(USB_SERIAL_BAUDRATE);
  Serial.setTimeout(100);
  
  // 初始化UART1（用于与调试板通信）
  // 注意：ESP32的HardwareSerial.begin()参数顺序是：baudrate, config, RX, TX
  // 但有些版本可能是：baudrate, config, TX, RX，如果不行请尝试交换
  DebugBoardSerial.begin(DEBUG_BOARD_BAUDRATE, SERIAL_8N1, 
                         DEBUG_BOARD_SERIAL_RX, DEBUG_BOARD_SERIAL_TX);
  DebugBoardSerial.setTimeout(100);
  
  // 如果上面的顺序不对，尝试这个（交换RX和TX）：
  // DebugBoardSerial.begin(DEBUG_BOARD_BAUDRATE, SERIAL_8N1, 
  //                        DEBUG_BOARD_SERIAL_TX, DEBUG_BOARD_SERIAL_RX);
  
  // 等待UART稳定
  delay(500);
  
  // 验证UART是否初始化成功
  Serial.print("[UART] UART1 initialized: RX=");
  Serial.print(DEBUG_BOARD_SERIAL_RX);
  Serial.print(", TX=");
  Serial.println(DEBUG_BOARD_SERIAL_TX);
  
  // 等待串口就绪
  delay(1000);
  
  Serial.println("ESP32 Servo Controller Ready");
  Serial.println("Waiting for commands from Python...");
  Serial.print("Debug Board UART: RX=");
  Serial.print(DEBUG_BOARD_SERIAL_RX);
  Serial.print(", TX=");
  Serial.print(DEBUG_BOARD_SERIAL_TX);
  Serial.print(", Baudrate=");
  Serial.println(DEBUG_BOARD_BAUDRATE);
  
  // 显示配置信息
  Serial.print("[CONFIG] Add newline after command: ");
  Serial.println(ADD_NEWLINE_AFTER_COMMAND ? "YES" : "NO");
  if (ADD_NEWLINE_AFTER_COMMAND) {
    Serial.print("[CONFIG] Newline type: ");
    Serial.println(USE_CRLF ? "\\r\\n" : "\\n");
  }
  
  // 测试UART连接（发送测试命令）
  Serial.println("[TEST] Testing UART connection to debug board...");
  Serial.println("[TEST] Sending test command: #000P1500T1000!");
  
  // 发送测试命令
  for (int i = 0; i < 15; i++) {
    const char* testCmd = "#000P1500T1000!";
    DebugBoardSerial.write(testCmd[i]);
    delayMicroseconds(100);
  }
  DebugBoardSerial.flush();
  delay(200);
  
  Serial.println("[TEST] Test command sent. Check if debug board received it.");
  Serial.println("[TEST] If debug board responded, UART is working correctly.");
}

// ============================================
// Loop函数
// ============================================

void loop() {
  // 从USB串口（Python）读取数据
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    // 添加字符到缓冲区
    if (c >= 32 && c <= 126) {  // 可打印字符
      commandBuffer += c;
    }
    
    // 检查是否收到完整的多舵机指令（以}结尾）
    if (c == '}' && commandBuffer.length() > 0 && commandBuffer[0] == '{') {
      // 多舵机指令结束
      processCommand(commandBuffer);
      commandBuffer = "";
    }
    // 检查是否收到完整的单舵机指令（以!结尾且以#开头）
    else if (c == '!' && commandBuffer.length() > 0 && commandBuffer[0] == '#') {
      // 单舵机指令结束
      processCommand(commandBuffer);
      commandBuffer = "";
    }
    
    // 防止缓冲区溢出
    if (commandBuffer.length() > 200) {
      Serial.println("Error: Command buffer overflow");
      Serial.print("Buffer content: ");
      Serial.println(commandBuffer);
      commandBuffer = "";
    }
  }
  
  // 注意：暂时不读取调试板响应，专注于发送数据
  // 如果需要读取响应，取消下面的注释
  // while (DebugBoardSerial.available() > 0) {
  //   char c = DebugBoardSerial.read();
  //   Serial.write(c);  // 转发到USB串口
  // }
  
  delay(10);  // 短暂延迟，避免CPU占用过高
}

// ============================================
// 函数实现
// ============================================

bool parseServoCommand(String cmd, int& servoId, int& pwm, int& time) {
  // 格式: #000P1500T1000!
  // 检查基本格式
  if (cmd.length() < 13 || cmd[0] != '#' || cmd[cmd.length()-1] != '!') {
    return false;
  }
  
  // 查找P和T的位置
  int pPos = cmd.indexOf('P');
  int tPos = cmd.indexOf('T');
  
  if (pPos == -1 || tPos == -1 || pPos >= tPos) {
    return false;
  }
  
  // 解析ID (3位数字)
  String idStr = cmd.substring(1, pPos);
  servoId = idStr.toInt();
  
  // 解析PWM (4位数字)
  String pwmStr = cmd.substring(pPos + 1, tPos);
  pwm = pwmStr.toInt();
  
  // 解析TIME (4位数字，到!之前)
  String timeStr = cmd.substring(tPos + 1, cmd.length() - 1);
  time = timeStr.toInt();
  
  // 验证范围
  if (servoId < 0 || servoId > 254) return false;
  if (pwm < 500 || pwm > 2500) return false;
  if (time < 0 || time > 9999) return false;
  
  return true;
}

bool parseMultiServoCommand(String cmd) {
  // 格式: {#000P1500T1000!#001P2500T0000!}
  if (cmd.length() < 3 || cmd[0] != '{' || cmd[cmd.length()-1] != '}') {
    return false;
  }
  
  // 移除外层{}
  String innerCmd = cmd.substring(1, cmd.length() - 1);
  
  // 分割多个指令（以!分隔）
  int startPos = 0;
  bool allValid = true;
  
  while (startPos < innerCmd.length()) {
    int endPos = innerCmd.indexOf('!', startPos);
    if (endPos == -1) break;
    
    String singleCmd = innerCmd.substring(startPos, endPos + 1);
    int servoId, pwm, time;
    
    if (!parseServoCommand(singleCmd, servoId, pwm, time)) {
      allValid = false;
      Serial.print("Error parsing command: ");
      Serial.println(singleCmd);
    } else {
      Serial.print("Parsed: Servo ");
      Serial.print(servoId);
      Serial.print(", PWM=");
      Serial.print(pwm);
      Serial.print(", TIME=");
      Serial.println(time);
    }
    
    startPos = endPos + 1;
  }
  
  return allValid;
}

void sendToDebugBoard(String command) {
  // 发送指令到调试板
  // 注意：这里直接转发原始指令
  // 如果调试板需要不同的格式，需要在这里转换
  
  Serial.print("[DEBUG] Sending to debug board: ");
  Serial.println(command);
  Serial.print("[DEBUG] Command length: ");
  Serial.println(command.length());
  
  // 检查UART是否可用
  if (!DebugBoardSerial) {
    Serial.println("[ERROR] UART not initialized!");
    return;
  }
  
  // 显示发送前的UART状态
  Serial.print("[DEBUG] UART TX buffer before send: ");
  Serial.println(DebugBoardSerial.availableForWrite());
  
  // 直接发送整个命令字符串，不使用逐字节延迟
  // 这样可以更快地发送，避免调试板超时
  Serial.print("[DEBUG] Sending bytes: ");
  for (int i = 0; i < command.length(); i++) {
    Serial.print((int)command[i], HEX);
    Serial.print(" ");
  }
  Serial.println();
  
  // 一次性发送整个命令，不使用延迟
  DebugBoardSerial.print(command);
  
  // 如果调试板需要换行符来识别命令结束（当前设置为不需要）
  if (ADD_NEWLINE_AFTER_COMMAND) {
    if (USE_CRLF) {
      DebugBoardSerial.write('\r');
      DebugBoardSerial.write('\n');
      Serial.println("[DEBUG] Added \\r\\n after command");
    } else {
      DebugBoardSerial.write('\n');
      Serial.println("[DEBUG] Added \\n after command");
    }
  }
  
  // 强制刷新UART缓冲区，确保数据立即发送
  DebugBoardSerial.flush();
  
  // 最小延迟，只等待UART硬件发送完成
  // 115200波特率，15字节大约需要1.3ms，但为了安全起见等待5ms
  delay(5);
  
  // 验证发送 - 检查UART状态
  Serial.print("[DEBUG] UART TX buffer after send: ");
  Serial.println(DebugBoardSerial.availableForWrite());
  
  // 验证所有数据是否已发送
  if (DebugBoardSerial.availableForWrite() == 128) {
    Serial.println("[DEBUG] All data flushed from TX buffer");
  } else {
    Serial.print("[WARNING] TX buffer still has data: ");
    Serial.println(128 - DebugBoardSerial.availableForWrite());
  }
  
  Serial.println("[DEBUG] Command sent successfully to debug board");
  
  // 不等待，立即返回，让调试板自己处理
  // 如果调试板需要处理时间，可以在调用此函数后添加延迟
}

void processCommand(String command) {
  Serial.println("========================================");
  Serial.print("[ESP32] Received command: ");
  Serial.println(command);
  Serial.print("[ESP32] Command length: ");
  Serial.println(command.length());
  
  // 检查是多舵机指令还是单舵机指令
  if (command.length() == 0) {
    Serial.println("[ESP32] Error: Empty command");
    return;
  }
  
  if (command[0] == '{') {
    // 多舵机指令
    Serial.println("[ESP32] Detected multi-servo command");
    if (parseMultiServoCommand(command)) {
      Serial.println("[ESP32] Command parsed successfully");
      sendToDebugBoard(command);
      Serial.println("[ESP32] Multi-servo command executed");
    } else {
      Serial.println("[ESP32] Error: Invalid multi-servo command format");
    }
  } else if (command[0] == '#') {
    // 单舵机指令
    Serial.println("[ESP32] Detected single-servo command");
    int servoId, pwm, time;
    if (parseServoCommand(command, servoId, pwm, time)) {
      Serial.println("[ESP32] Command parsed successfully");
      sendToDebugBoard(command);
      Serial.print("[ESP32] Single-servo command executed: Servo ");
      Serial.print(servoId);
      Serial.print(", PWM=");
      Serial.print(pwm);
      Serial.print(", TIME=");
      Serial.println(time);
    } else {
      Serial.println("[ESP32] Error: Invalid servo command format");
    }
  } else {
    Serial.print("[ESP32] Error: Unknown command format. First char: ");
    Serial.println((int)command[0]);
  }
  Serial.println("========================================");
}
