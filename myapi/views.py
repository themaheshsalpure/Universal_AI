from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
# import json
from huggingface_hub import login

huggingface_token = "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"
login(token=huggingface_token)

# Initialize the Hugging Face model and tokenizer
model = AutoModelForCausalLM.from_pretrained("pansophic/rocket-3B", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("pansophic/rocket-3B", trust_remote_code=True, torch_dtype=torch.bfloat16)
streamer = TextStreamer(tokenizer)

def home(request):
    return HttpResponse("API")

@csrf_exempt
@require_GET
def get_response(request, user_message):
    user_message = str(user_message).lower()
    prompt = """system
    {system}
    user
    {user}
    assistant
    """

    system = "As a revered wellness guru, you unveil the ancient secrets of traditional Indian cures for common ailments, offering profound insights and guidance. Share your wisdom one remedy at a time, infusing each response with the time-honored essence of holistic health practices"
    user = user_message

    # Apply the ChatML format
    prompt = prompt.format(system=system, user=user)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to("cuda")
    generated_text = model.generate(**inputs, max_length=200, top_p=0.95, do_sample=True, temperature=0.7, use_cache=True, streamer=streamer)

    generated_text_ids = generated_text[0].tolist()
    reply = tokenizer.decode(generated_text_ids, skip_special_tokens=True)

    model_response = {'response': reply}
    return JsonResponse(model_response)

@csrf_exempt
@require_POST
def make_appointment(request, user_message):
    user_message = user_message.lower()
    if user_message.count("book") >= 1 or user_message.count("appointment") >= 1:
        user_message = "Select nearby hospitals to book an appointment like Shantanu Hospital, Rahul Hospital, Sanket Hospital, or Shreyash Hospital"
    elif user_message.count("shantanu") >= 1:
        user_message = "7083694063"
    elif user_message.count("mahesh") >= 1:
        user_message = "9403512671"
    elif user_message.count("sanket") >= 1:
        user_message = "7083694063"
    elif user_message.count("shreyash") >= 1 or user_message.count("shreyas") >= 1:
        user_message = "9403512671"
    else:
        user_message = "Please choose a hospital first."

    model_response = {'response': user_message}
    return JsonResponse(model_response)
