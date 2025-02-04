using ImageRecognition.Models.Test;

namespace ImageRecognition.Services.Interfaces
{
    public interface ITestService
    {
        /// <summary>
        ///    Get weather forecasts
        /// </summary>
        /// <returns> List of Weather Forecasts </returns>
        List<WeatherForecast> GetWeatherForecasts();
    }
}
