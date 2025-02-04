namespace ImageRecognition.Models.Output
{
    public class TrainingOutput
    {
        public float[][] weights { get; set; }
        public float[] biases { get; set; }
        public string[] classLabels { get; set; }
    }
}
