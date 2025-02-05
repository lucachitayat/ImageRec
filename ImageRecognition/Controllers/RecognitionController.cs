using ImageRecognition.Models.Test;
using ImageRecognition.Services;
using ImageRecognition.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace ImageRecognition.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class RecognitionController : ControllerBase
    {
        private readonly ILogger<TestController> _logger;
        private readonly IRecognitionService _recogService;

        public RecognitionController(ILogger<TestController> logger, IRecognitionService recogService)
        {
            _logger = logger;
            _recogService = recogService;
        }

        [HttpGet]
        public void Get()
        {
            return;
        }

        [HttpPost]
        public IActionResult Post()
        {
            _recogService.GetResNetClassifierModel();

            return Ok();
        }
    }
}
