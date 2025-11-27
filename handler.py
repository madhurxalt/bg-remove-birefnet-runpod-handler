import runpod
from app import remove_background_url

def handler(job):
    job_input = job["input"]
    input_image_url = job_input.get("image_url", "")
    
    result = remove_background_url(input_image_url)
    
    return result

runpod.serverless.start({"handler": handler})