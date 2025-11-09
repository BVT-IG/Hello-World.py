using Microsoft.UI.Xaml.Controls;  // 控件：按钮、文本框
using System;  // 系统基础：异常、字符串
using System.IO;  // 文件操作
using System.Threading.Tasks;  // 异步操作，防 UI 卡顿
using Microsoft.ML.OnnxRuntimeGenAI;  // 最新 GenAI 库（0.9.0），简化生成循环
using Microsoft.ML.OnnxRuntime;  // ONNX Runtime 核心（1.23.2）

namespace TextGenAIUpdated
{
    public sealed partial class MainWindow : Window
    {
        private Model model;  // ONNX 模型对象（新包加载）
        private string conversationHistory = "";  // 存储对话历史

        public MainWindow()
        {
            this.InitializeComponent();  // 初始化 UI
            LoadModelAsync();  // 启动时异步加载模型
        }

        // 异步加载模型（使用最新 GenAI + DirectML GPU）
        private async void LoadModelAsync()
        {
            try
            {
                // 步骤1: 加载 ONNX 模型（替换为你的模型路径）
                string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model.onnx");  // 模型路径
                model = new Model(modelPath);  // 使用 GenAI 的 Model 类加载

                // 步骤2: 配置 GPU 加速（DirectML 执行提供者）
                var options = new InferenceOptions();  // 推理选项（新 API）
                options.SetExecutionProvider("DirectML", 0);  // DirectML GPU（0 是设备 ID，默认主显卡）
                // 如果无 GPU，可添加：options.SetExecutionProvider("CPU", 0);

                // 步骤3: 创建生成器并配置参数（适用于 Phi-3 等文本生成模型）
                var generator = new Generator(model, options);  // 生成器实例
                generator.SetMaxLength(512);  // 最大生成 512 token（约 300-400 字）
                generator.SetTemperature(0.7f);  // 温度 0.7（平衡创意和准确）
                generator.SetTopP(0.9f);  // Top-P 采样，控制多样性
                generator.SetDoSample(true);  // 启用随机采样（非贪婪生成）

                // GenAI 内置 tokenizer 支持，如果需自定义：var tokenizer = new Tokenizer("tokenizer.json");

                OutputTextBlock.Text = "模型加载成功（使用最新 ONNX Runtime 1.23.2 + DirectML GPU）！输入提示开始对话。";
                GenerateButton.IsEnabled = true;  // 启用生成按钮
            }
            catch (Exception ex)
            {
                OutputTextBlock.Text = "加载失败: " + ex.Message + "\n检查模型路径、GPU 驱动或包版本。";
            }
        }

        // 生成对话按钮点击事件（纯文本生成）
        private async void GenerateText_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(PromptTextBox.Text) || model == null)
            {
                OutputTextBlock.Text = "请输入提示或检查模型加载！";
                return;
            }

            ProgressBar.Visibility = Visibility.Visible;  // 显示进度条
            GenerateButton.IsEnabled = false;  // 禁用按钮防重复点击

            try
            {
                // 步骤1: 获取用户输入并添加到对话历史
                string userPrompt = PromptTextBox.Text.Trim();  // 用户输入
                conversationHistory += $"\n用户: {userPrompt}\n";  // 记录历史（上下文）

                // 步骤2: 编码输入为 token（GenAI 内置编码，含历史）
                var tokens = model.Encode(conversationHistory);  // Encode 整个对话（数字序列）

                // 步骤3: 运行生成循环（新 GenAI API，自动处理 autoregressive）
                var options = new InferenceOptions { ExecutionProvider = "DirectML" };  // 复用 GPU
                var generator = new Generator(model, options);  // 新生成器实例
                var outputTokens = generator.GenerateTokens(tokens, 100);  // 生成 100 个新 token（可调）

                // 步骤4: 解码输出为文本
                string generatedText = model.Decode(outputTokens);  // Decode token 转文字

                // 步骤5: 更新对话历史和 UI
                conversationHistory += $"AI: {generatedText}\n";  // 追加 AI 回复
                OutputTextBlock.Text = conversationHistory;  // 显示完整历史
                PromptTextBox.Text = "";  // 清空输入框
            }
            catch (Exception ex)
            {
                OutputTextBlock.Text += $"\n生成失败: {ex.Message}\n检查输入长度或 GPU 内存（建议 >4GB）。";
            }
            finally
            {
                ProgressBar.Visibility = Visibility.Collapsed;  // 隐藏进度条
                GenerateButton.IsEnabled = true;  // 恢复按钮
            }
        }
    }
}