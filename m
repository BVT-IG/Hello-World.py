using System;  // 系统基础库
using Microsoft.ML.OnnxRuntime;  // ONNX Runtime核心
using Microsoft.ML.OnnxRuntimeGenAI;  // GenAI扩展，用于生成循环

class Program
{
    static void Main(string[] args)
    {
        // 步骤1: 指定模型文件夹路径（替换成你的D盘路径，例如D:\gemma2，确保文件夹含model.onnx、genai_config.json、tokenizer.model）
        string modelDir = @"D:\gemma2";  // Gemma-2 ONNX文件夹路径

        try
        {
            // 步骤2: 配置SessionOptions（启用GPU加速）
            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_DirectML(0);  // 使用DirectML GPU（0是设备ID）；如果无GPU，注释此行或改成AppendExecutionProvider_CPU(0)

            // 步骤3: 加载模型（Model构造函数加载整个文件夹）
            using (var model = new Model(modelDir, sessionOptions))  // 如果无GPU选项，用new Model(modelDir)
            {
                // 步骤4: 加载Tokenizer（Gemma-2通常用tokenizer.model；如果你的文件是tokenizer.json，用new Tokenizer(Path.Combine(modelDir, "tokenizer.json")))
                using (var tokenizer = new Tokenizer(model))  // 自动从文件夹加载tokenizer.model
                {
                    // 步骤5: 准备输入提示（简单文本）
                    string prompt = "Hello, how are you today?";  // 测试提示；你可以改成Console.ReadLine()用户输入

                    // 编码提示为token（数字序列）
                    var tokens = tokenizer.Encode(prompt);

                    // 步骤6: 配置生成参数
                    using (var generatorParams = new GeneratorParams(model))
                    {
                        generatorParams.SetInputSequences(tokens);  // 设置输入token
                        generatorParams.TryGraphCaptureWithMaxBatchSize(1);  // 优化批处理（1表示单条输入）
                        generatorParams.SetSearchOption("max_length", 100);  // 最大生成长度（token数，约50-70字）
                        generatorParams.SetSearchOption("temperature", 0.7f);  // 温度（创意度，0-1）
                        generatorParams.SetSearchOption("top_p", 0.9f);  // Top-P采样（多样性）

                        // 步骤7: 创建生成器并运行推理循环
                        using (var generator = new Generator(model, generatorParams))
                        {
                            Console.WriteLine("生成输出：");

                            while (!generator.IsDone())  // 循环生成直到结束
                            {
                                generator.ComputeLogits();  // 计算logits（概率）
                                generator.GenerateNextToken();  // 生成下一个token

                                // 获取最新token并解码为文本
                                var outputTokens = generator.GetSequence(0);  // 获取输出序列
                                var newToken = outputTokens[^1];  // 最后一个token
                                string outputText = tokenizer.Decode(newToken);  // 解码

                                Console.Write(outputText);  // 逐字输出（流式）
                            }

                            Console.WriteLine();  // 换行
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("错误: " + ex.Message);  // 错误处理，例如路径错或GPU不支持
        }

        Console.WriteLine("按任意键退出...");
        Console.ReadKey();
    }
}