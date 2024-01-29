import azure.functions as func
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import os
import openai
import time
import logging

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="voicelens", methods=['POST'])
def voicelens(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    message = req.params.get('message')
    if not message:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            image = req_body.get('image_url')
            # image_text = extract_text_from_image(image)
            message = req_body.get('message')
            audio_data = synthesize_speech_to_audio_data(message)

    if message:
        return func.HttpResponse(audio_data)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )

def extract_text_from_image(image_url):
    """
    Extracts text from the given image URL using the Azure Computer Vision API.

    Args:
    image_url (str): The URL of the image from which to extract the text.

    Returns:
    str: Extracted text from the image or None if the extraction fails.
    """
    # Authenticate
    subscription_key = os.environ["MULTI_SERVICE_KEY"]
    endpoint = os.environ["MULTI_SERVICE_ENDPOINT"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    '''
    OCR: Read File using the Read API, extract text - remote
    This example will extract text in an image, then print results, line by line.
    This API call can also extract handwriting style text (not shown).
    '''

    # Call API with URL and raw response (allows you to get the operation location)
    read_response = computervision_client.read(image_url,  raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        # Output the result
        extracted_text = "".join(
            line.text for text_result in read_result.analyze_result.read_results
            for line in text_result.lines
        )
    return extracted_text

def get_chat_completion(message_text):
    # Configure the OpenAI API settings
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_API_BASE")
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    _messages =  [
        {
            "role": "system",
            "content": ("Below is the text read by OCR. The following is an "
                        "introduction to the Japanese text. Please check the "
                        "content and make sure the text is appropriate. If it is "
                        "not correct, please complete it so that it can be read out. "
                        "・Don't change the numerical values in the sentences.")
        },
        {
            "role": "user",
            "content": message_text
        }
    ]
    # Process the message using the Chat API
    completion = openai.ChatCompletion.create(
        engine=os.getenv("OPENAI_API_ENGINE"),
        messages=_messages,
        temperature=0.7,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion['choices'][0]['message']['content']

def synthesize_speech_to_audio_data(text):
    """
    Converts the given text to speech and returns the audio data.

    Args:
    text (str): Text to be converted to speech.

    Returns:
    bytes: The audio data in binary format.
    """

    # Configure Azure Speech Service
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('MULTI_SERVICE_KEY'), 
                                           region=os.environ.get('MULTI_SERVICE_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # Set the voice for speech synthesis
    speech_config.speech_synthesis_voice_name='ja-JP-KeitaNeural'

    # Create an instance of SpeechSynthesizer
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, 
                                                     audio_config=audio_config)

    # Convert text to speech
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    # If synthesis is successful, return the audio data
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return speech_synthesis_result.audio_data

    # If synthesis fails, raise an exception
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        raise Exception(f"Speech synthesis canceled: {cancellation_details.reason}, "
                        f"Error details: {cancellation_details.error_details}")