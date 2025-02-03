using ImageRecognition.Models.Test;
using ImageRecognition.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace ImageRecognition.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class TestController : ControllerBase
    {
        private readonly ILogger<TestController> _logger;
        private readonly ITestService _testService;

        public TestController(ILogger<TestController> logger, ITestService testService)
        {
            _logger = logger;
            _testService = testService;
        }

        [HttpGet(Name = "GetWeatherForecast")]
        public IEnumerable<WeatherForecast> Get()
        {
            return _testService.GetWeatherForecasts();
        }
    }
}
