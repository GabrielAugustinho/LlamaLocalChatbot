using LLama;
using LLama.Common;
using LLama.Native;

namespace consoleapp_llamasharp
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            Program p = new Program();

            // Go to the website https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main download the llama-2-7b-chat file. Q5_K_M.gguf AND PUT IT IN THE MODELS FOLDER
            string smodelpath = Path.Combine("..", "..", "..", "Models", "READ THE REVIEW");

            // For the chat have its characteristics, change the prompt
            var prompt = "Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, " +
                         "and never fails to answer the User's requests immediately and with precision.\\r\\n\\r\\n" +
                         "User: Hello, Bob.\\r\\n" +
                         "Bob: Hello. How may I help you today?\\r\\n" +
                         "User: Please tell me the largest city in Europe.\\r\\n" +
                         "Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.\\r\\n" +
                         "User:";
            NativeLibraryConfig.Instance.WithLogs();

            // Load a model
            var parameters = new ModelParams(smodelpath)
            {
                ContextSize = 16384,
                GpuLayerCount = 4,
                UseMemoryLock = true,
                UseMemorymap = true
            };
            using var model = LLamaWeights.LoadFromFile(parameters);

            // Initialize a chat session
            using var context = model.CreateContext(parameters);
            var ex = new InteractiveExecutor(context);
            ChatHistory ch = new();
            ch.AddMessage(AuthorRole.System, "");
            ChatSession session = new(ex, ch);

            // show the prompt
            Console.WriteLine();
            Console.Write(prompt);

            // run the inference in a loop to chat with LLM
            while (prompt != "stop")
            {
                prompt = Console.ReadLine() ?? "";
                await foreach (
                var text in session.ChatAsync(
                new ChatHistory.Message(AuthorRole.User, prompt),
                new InferenceParams
                {
                    Temperature = 0.1f,
                    AntiPrompts = ["\r\nUser: "],
                    TopK = 50,
                    TopP = 0.6f,
                    RepeatPenalty = 1.1f
                }))
                {
                    Console.Write(text);
                }
            }

            // if you want, save the session into a (.txt) file via StreamWriter:
            //session.SaveSession(pathto_txt_filename);
        }

    }
}