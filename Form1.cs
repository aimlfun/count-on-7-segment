using System.Diagnostics;

namespace Count
{
    /// <summary>
    /// You're thinking "just why, Dave?". Because I can.
    /// 
    /// This demo shows how one can teach AI to sum() all inputs regardless of order
    /// and then display on a 7 segment display.
    /// 
    /// The AI training associates a number with the segments in the "7 segment display" to illuminate, 
    /// to make it more complicated (otherwise it's just a DAC).
    /// 
    /// Clever? Debateable. I wanted to prove that way to solve the count(), but also with an 
    /// answer beyond a simple value.
    /// 
    /// Sure you don't need AI to achieve this. It can be done in simple code, however I
    /// think it's a legitimate demo of the association of input to a more complex output.
    /// </summary>
    public partial class Form1 : Form
    {
        /// <summary>
        /// Where it will read / write the AI model file.
        /// </summary>
        private const string c_aiModelFilePath = @"c:\temp\count.ai";

        /// <summary>
        /// Our NN.
        /// </summary>
        readonly NeuralNetwork neuralNetwork;

        /// <summary>
        /// Constructor.
        /// </summary>
        public Form1()
        {
            InitializeComponent();

            // input:  9 values, each 1 or 0. If "1" then it "counts" it. i.e. AI is working out SUM(input)
            // output: 7 values, each approximately 1 or 0 representing segments of the LED. If "1"-ish (>0.8) lights the segment.
            int[] layers = new int[6] { 9           /* INPUT: items to count 0-9 */,
                                        9, 9, 9, 9, /* HIDDEN (I have no idea what the optimal might be, but 4x9 works) */
                                        7           /* OUTPUT: segments to light */ };

            // TanH is my preferred function, so far I've seen limited potential in the others.
            ActivationFunctions[] activationFunctions = new ActivationFunctions[6] { ActivationFunctions.TanH, ActivationFunctions.TanH, ActivationFunctions.TanH,
                                                                                     ActivationFunctions.TanH, ActivationFunctions.TanH, ActivationFunctions.TanH };

            neuralNetwork = new(0, layers, activationFunctions, false);
        }

        /// <summary>
        /// When the form loads we train the AI.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            // disable the checkboxes, until we have trained the model.
            SetCheckBoxEnableState(false);

            Show(); // training may take a while, show the UI whilst training so they are not wondering what's happening.

            Train(); // train it to map a number to the respective segments.

            AskAItoCount(); // should show "0" (as no checkboxes have been "checked")

            // enable the checkboxes that we count, so user can try different numbers.
            SetCheckBoxEnableState(true);
        }

        /// <summary>
        /// Finds the checkboxes and disables or enables them. Used to stop users clicking whilst
        /// it's training.
        /// </summary>
        /// <param name="value"></param>
        private void SetCheckBoxEnableState(bool value)
        {
            foreach (Control c in Controls) if (c is CheckBox checkbox) checkbox.Enabled = value;
        }

        /// <summary>
        /// Trains the AI to count, and illuminate the segments.
        /// </summary>
        private void Train()
        {
            // load a pre-trained model if found.
            if (File.Exists(c_aiModelFilePath))
            {
                neuralNetwork.Load(c_aiModelFilePath);
                Text = "7 Segment Display Count via AI - MODEL LOADED"; // we know it passed training
                return;
            }

            // no pre-trained model, let's train it.

            /* 7 segments are annotated as follows:
             * 
             *      [-a-]                 
             *   [|]     [|]
             *   [f]     [b]
             *   [|]     [|]
             *      [-g-]              
             *   [|]     [|]
             *   [e]     [c]            
             *   [|]     [|]
             *      [-d-]              
             */

            // HOW DOES TRAINING WORK?

            // We tell the AI if the input is 0,0,0,0,0,0,0,0,1 the count is 1 (one "1" in the array), segments to light are b & c.
            // We tell the AI if the input is 0,0,0,0,0,0,0,1,0 the count is also 1, segments to light are b & c.
            // We tell the AI if the input is 0,1,0,0,0,0,0,0,0 the count is again 1, segments to light are b & c. i.e. it doesn't matter which are "1", just the count.
            // We tell the AI if the input is 0,1,0,1,0,1,0,0,0 the count is 3 (there are three "1"s), the segments to light are a,b,c,d & g.
            // Etc. (We train for all permutations 000000000 to 111111111).

            // The AI learns the association of a known "input" with a desired "output". To find the sweet spot for
            // all, we use back-propagation.

            // e.g. if the count is a "1" we light segments "a"-"f" but not "g" (g would make "0" an "8").
            double[][] segmentsToLight = {
                           // a,b,c,d,e,f,g
                new double[]{ 1,1,1,1,1,1,0}, // 0
                new double[]{ 0,1,1,0,0,0,0}, // 1
                new double[]{ 1,1,0,1,1,0,1}, // 2
                new double[]{ 1,1,1,1,0,0,1}, // 3
                new double[]{ 0,1,1,0,0,1,1}, // 4
                new double[]{ 1,0,1,1,0,1,1}, // 5
                new double[]{ 1,0,1,1,1,1,1}, // 6
                new double[]{ 1,1,1,0,0,0,0}, // 7
                new double[]{ 1,1,1,1,1,1,1}, // 8
                new double[]{ 1,1,1,1,0,1,1}  // 9
            };

            bool trained = false;

            // train the AI up to 50k times, exiting if we are getting correct answers.
            // note: it could fail even with 50k, simply because it picks bad initial weights and biases.
            for (int i = 0; i < 50000; i++)
            {
                // 0.511 = 2^9 (permutation 000000000 to 111111111).
                for (int n = 0; n < 512; n++)
                {
                    double[] inputs = BinaryAmount(n, out int count);

                    // associate the inputs with a 7 segment display
                    neuralNetwork.BackPropagate(inputs, segmentsToLight[count] /* segments to light based on number of "1"s in input */);
                }

                // by this point we *may* have done enough training...
                // we check the output, and if it's accurate, we exit. We don't check before 15k,
                // because it would slow training down for little gain.
                if (i > 15000)
                {
                    trained = true;

                    // test the result for all permutations, and if all are good, we're trained.
                    for (int n = 0; n < 512; n++)
                    {
                        double[] inputs = BinaryAmount(n, out int count);

                        if (!IsTrained(neuralNetwork.FeedForward(inputs), segmentsToLight[count]))
                        {
                            trained = false; // wrong answer
                            break;
                        }
                    }

                    if (trained)
                    {
                        Text = "7 Segment Display Count via AI - TRAINED."; // we know it passed training
                        neuralNetwork.Save(c_aiModelFilePath);
                        break;
                    }
                }

                if (i % 1000 == 0) // indicator of progress, every 1000
                {
                    Text = $"7 Segment Display Count via AI - TRAINING. GENERATION {i}";
                    Application.DoEvents();
                }
            }

            // Remember back propagation is about finding the right weights/biases to provide the desired outcome. We pick the initial value
            // at random, and sometimes we start from a bad initial weights that takes much more iterations. Rather than try forever, we give
            // up at 50k, and ask the user to re-run.
            if (!trained) MessageBox.Show("Unable to train successfully (poor initial random weights/biases), please re-run.");
        }

        /// <summary>
        /// See if all the AI outputs match the desired output.
        /// </summary>
        /// <param name="aiOutput">Output of the AI.</param>
        /// <param name="desiredOutput">What the AI *should* have output.</param>
        /// <returns>True - training resulted in correct response | False - needs more training.</returns>
        private static bool IsTrained(double[] aiOutput, double[] desiredOutput)
        {
            for (int outputIndex = 0; outputIndex < aiOutput.Length; outputIndex++)
            {
                int value1 = (int)Math.Round(aiOutput[outputIndex]); // remember NN's don't give PRECISE results, so we have to round.
                int value2 = (int)desiredOutput[outputIndex];

                if (value1 != value2) return false; // one doesn't match, no point in checking further...
            }

            return true; // all match
        }

        /// <summary>
        /// Returns 1 / 0 in each output cell based on the input number. *We use it to train AI.
        /// </summary>
        /// <param name="inputValue"></param>
        /// <param name="countOf1sInBinaryRepresentationOfInputValue">How many 1's appear in the the inputValue when converted to "binary".</param>
        /// <returns>Array of 1's & zero's from the binary representation of the input value.</returns>
        private static double[] BinaryAmount(int inputValue, out int countOf1sInBinaryRepresentationOfInputValue)
        {
            string binaryAmount = Convert.ToString(inputValue, 2).PadLeft(9, '0'); // turn the input into binary

            double[] result = new double[binaryAmount.Length];

            countOf1sInBinaryRepresentationOfInputValue = 0;

            for (int digitIndex = 0; digitIndex < binaryAmount.Length; digitIndex++)
            {
                result[digitIndex] = float.Parse(binaryAmount[digitIndex].ToString());

                if (result[digitIndex] != 0) ++countOf1sInBinaryRepresentationOfInputValue; // count the "1"s
            }

            return result; // how many "1"s in the binary string
        }

        /// <summary>
        /// As you tick/untick, we ask the AI to count how many boxes are checked.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void CheckBox_CheckedChanged(object sender, EventArgs e)
        {
            AskAItoCount();
        }

        /// <summary>
        /// Magic that uses feedforward to determine which segments to light.
        /// </summary>
        private void AskAItoCount()
        {
            double[] inputs =
            {
                BoolTo1or0(checkBox1.Checked),
                BoolTo1or0(checkBox2.Checked),
                BoolTo1or0(checkBox3.Checked),
                BoolTo1or0(checkBox4.Checked),
                BoolTo1or0(checkBox5.Checked),
                BoolTo1or0(checkBox6.Checked),
                BoolTo1or0(checkBox7.Checked),
                BoolTo1or0(checkBox8.Checked),
                BoolTo1or0(checkBox9.Checked)
            };

            double[] outputOfAIelementsAtoG = neuralNetwork.FeedForward(inputs);

            richTextBoxAIOutput.Text = $"AI INPUT: (CHECKBOXES)\r\n" +
                                       $"{string.Join("\r\n", inputs)}\r\n"+
                                       "\r\n" +
                                       $"AI OUTPUT: (SEGMENTS TO ILLUMINATE)\r\n" +
                                       $"{AIOutputDecodedAsSegmentsToIlluminate(outputOfAIelementsAtoG)}";

            pictureBox7Segment.Image?.Dispose();
            pictureBox7Segment.Image = SevenSegmentDisplay.Output(outputOfAIelementsAtoG);
        }

        /// <summary>
        /// Show the AI output interpreted.
        /// </summary>
        /// <param name="outputOfAIelementsAtoG"></param>
        /// <returns>1 row per AI output, with on/off and the neuron output.</returns>
        private static string AIOutputDecodedAsSegmentsToIlluminate(double[] outputOfAIelementsAtoG)
        {
            string result = "";

            for (int i = 0; i < outputOfAIelementsAtoG.Length; i++)
            {
                result += $"{Char.ConvertFromUtf32(97 + i)} {(outputOfAIelementsAtoG[i] > 0.8 ? "ON " : "OFF")} {outputOfAIelementsAtoG[i]:0.0000}\r\n";
            }

            return result;
        }

        /// <summary>
        /// Convert bool to 1 (true) / 0 (false).
        /// </summary>
        /// <param name="value">true/false.</param>
        /// <returns>1 if value==true | 0 if value == false</returns>
        private static double BoolTo1or0(bool value)
        {
            return value ? 1 : 0;
        }
    }
}