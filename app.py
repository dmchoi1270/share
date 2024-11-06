
from flask import Flask, request, jsonify, render_template, session
import openai
import os
from dotenv import load_dotenv
import json
import time
import sqlite3
import traceback
import threading
from datetime import datetime
import logging
import sys
from openai import OpenAI  # OpenAI API 클라이언트

# STT 설정 부분
import speech_recognition as sr
from pydub import AudioSegment

# Llama API
from llamaapi import LlamaAPI

# Claude API
import anthropic  # Import for Claude API
import re

# 환경 변수 로드
load_dotenv()

# 현재 파일의 디렉토리 경로 설정
# 개발 중에는 현재 파일의 경로를 사용
base_path = os.getcwd()

app = Flask(__name__, template_folder=f"{base_path}\\templates")

# 디버그 모드 설정
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# AI 설정
current_ai = os.getenv('CURRENT_AI')

# Thread, run 설정
thread = None

# 현재 설정된 AI 표시 
print(f"※  Current activation AI is {current_ai} !!!")

if current_ai == "chatgpt" :
    openai.api_key = os.getenv('OPENAI_API_KEY')
    assistant_id = os.getenv('ASSISTANTS_KEY')
    client = openai.OpenAI(api_key=openai.api_key)
    thread = client.beta.threads.create()
    thread_id = thread.id
    # run = client.beta.threads.runs.create(
    #         thread_id=thread.id,
    #         assistant_id=assistant_id
    # )
    # 스레드와 Run 생성
    # thread, run = create_thread_and_run()    
    if DEBUG_MODE:
        logging.info("Current thread with ID: %s", thread.id)
elif current_ai == "llama" :
    llama_api_key = os.getenv('LLAMA_API_KEY')  # Llama API key from environment
    # client = LlamaAPI(llama_api_key)
    client = OpenAI(
        api_key = llama_api_key,
        base_url = "https://api.llama-api.com"
    )
elif current_ai == "claude" :
    claude_api_key = os.getenv('ANTHROPIC_API_KEY')  # Claude API key from environment
    client = anthropic.Anthropic(api_key=claude_api_key)
elif current_ai == "hcnllama" :
    client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='hcn_ai_small'
    )
else:
    # 일반적인 상담 지식 질문에 대한 처리
    if DEBUG_MODE:
        print(f"You have no setting AI, Please check the environment file.")

# 세션별로 스레드를 관리하기 위한 잠금 객체 딕셔너리
# thread_locks = {}
# thread_locks_lock = threading.Lock()

# FFmpeg 경로 설정 (환경 변수로 설정)
ffmpeg_path = os.getenv('FFMPEG_PATH', 'ffmpeg')
AudioSegment.ffmpeg = ffmpeg_path

# 로깅 설정
logging.basicConfig(filename='app.log', level=logging.INFO if not DEBUG_MODE else logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/welcome', methods=['GET'])
def welcome_message():
    """웰컴 메시지를 반환하는 엔드포인트"""
    welcome_text = "안녕하세요. HCN 금호방송 기술지원 AI ▷금호아이◁ 입니다~^^ 무엇을 도와드릴까요?"
    return jsonify({'reply': welcome_text})

@app.route('/chat', methods=['POST'])
def chat():
    """사용자의 채팅 메시지를 처리하는 엔드포인트"""
    if DEBUG_MODE:
        logging.info("Start chat time: %s", datetime.now())
    user_message = request.json.get('message')

    if DEBUG_MODE:
        logging.info("User message: %s", user_message)
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Received user message: {user_message}")

    if not user_message:
        if DEBUG_MODE:
            logging.warning("Empty user message received.")
        return jsonify({'error': "Empty message received."}), 400

    try:
        if current_ai == "chatgpt" : # chatgpt AI
            
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_message
            )

            run = client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant_id
            )

            # 작업이 완료될 때까지 대기
            run = wait_on_run(run, thread)

            # Print for debugging
            if DEBUG_MODE:
                print(f"Run status after waiting: {run.status}")

            # Run에서 필요한 작업이 있는 경우 처리
            if run.required_action:
                if DEBUG_MODE:
                    print(f"Required action detected, processing tool calls.")

                reply = process_required_actions(run, thread)

                # 최종 봇 응답을 가져옴
                bot_reply = get_final_response(thread)

                deleted_message = client.beta.threads.messages.delete(
                    message_id=message.id,
                    thread_id=thread.id,
                )

                print(f"deleted_message : ",deleted_message )

                if DEBUG_MODE:
                    print(f"get_final_response: {bot_reply}")
            else:
                # 일반적인 상담 지식 질문에 대한 처리
                if DEBUG_MODE:
                    print(f"No required action, generating response from ChatGPT.")
                bot_reply = generate_response_from_chatgpt(thread, user_message)
        elif current_ai == "llama": # Llama AI
            # Generate response using Claude API
            bot_reply = generate_response_from_llama(user_message)
        elif current_ai == "claude": # Claude AI
            # Generate response using Claude API
            bot_reply = generate_response_from_claude(user_message)
        elif current_ai == "hcnllama": # Claude AI
            # Generate response using Claude API
            bot_reply = generate_response_from_hcnllama(user_message)
        else:
        # AI 점검 필요
            if DEBUG_MODE:
                print(f"You have no setting AI, Please check the environment file.")

        # 최종 봇 응답을 가져옴
        log_conversation(user_message, bot_reply)
        
        # Print for debugging
        if DEBUG_MODE:
            print(f"Final bot reply: {bot_reply}")
        
        return jsonify({'reply': bot_reply})

    except openai.error.InvalidRequestError as e:
        if DEBUG_MODE:
            logging.error(f"OpenAI Invalid Request Error: {str(e)}")
            print(f"Invalid Request Error: {str(e)}")  # Print for debugging
        return jsonify({'error': "Invalid request to the OpenAI service. Please try again later."}), 500

    except openai.error.OpenAIError as e:
        if DEBUG_MODE:
            logging.error(f"OpenAI Error: {str(e)}")
            print(f"OpenAI Error: {str(e)}")  # Print for debugging
        return jsonify({'error': "Error occurred with the OpenAI service. Please try again later."}), 500

    except Exception as e:
        if DEBUG_MODE:
            logging.error("Unexpected exception: %s", str(e))
            traceback.print_exc()
            print(f"Unexpected exception: {str(e)}")  # Print for debugging
        return jsonify({'error': "Unexpected error occurred. Please contact support."}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # 파일을 WAV로 변환
        wav_file_path = convert_to_wav(file)
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='ko-KR')
            os.remove(wav_file_path)  # 임시로 저장된 WAV 파일 삭제
            return jsonify({'text': text})
        except sr.UnknownValueError:
            return jsonify({'error': '음성을 인식할 수 없습니다.'})
        except sr.RequestError as e:
            return jsonify({'error': f'API 요청 오류: {e}'})    

# STT부분
def convert_to_wav(file):
    audio = AudioSegment.from_file(file)  # MP3나 다른 형식의 파일을 읽음
    wav_file = "temp.wav" 
    audio.export(wav_file, format="wav")  # WAV 파일로 변환
    return wav_file

def generate_response_from_chatgpt(thread, user_message):
    """ChatGPT API를 사용하여 답변을 생성하는 함수"""
    try:
        if DEBUG_MODE:
            print(f"Generating response for user message: {user_message}")

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )

        if DEBUG_MODE:
            print(f"Message to thread: {message}")
        
        if DEBUG_MODE:
            print(f"Sent user message to thread: {thread.id}")

        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            instructions="HCN의 방송 및 인터넷기술 운영 담장자의 입장에서 답변한다.",
        ) as stream:
            stream.until_done()

        if DEBUG_MODE:
            print(f"with client.beta.threads.runs.stream: {stream}")

        bot_reply = stream.get_final_messages()

        #bot_reply = stream.get_final_messages

        if DEBUG_MODE:
            print(f"with client.beta.threads.runs.stream: {bot_reply}")

        # 메시지 처리
        if bot_reply:
            bot_reply_texts = []
            for message in bot_reply:
                if message.content is not None:
                    if isinstance(message.content, list):
                        # message.content의 각 part에서 텍스트 추출
                        text_parts = [part.text.value.strip() for part in message.content if hasattr(part.text, 'value')]
                        bot_reply_texts.append(' '.join(text_parts))
                else:
                    # message.content 자체에서 텍스트 추출
                    if hasattr(message.content.text, 'value'):
                        bot_reply_texts.append(message.content.text.value.strip())

            bot_reply = ' '.join(bot_reply_texts)
        else:
            bot_reply = ""  # 빈 문자열로 초기화

        # 대화 내용을 파일에 기록
        log_conversation(user_message, bot_reply)
        
        if DEBUG_MODE:
            print(f"Generated ChatGPT reply: {bot_reply}")
        
        return bot_reply
    except Exception as e:
        if DEBUG_MODE:
            print("error : ", e)
            logging.error(f"Error while generating response from ChatGPT: {str(e)}")
            print(f"Error while generating response from ChatGPT: {str(e)}")  # Print for debugging
        return "죄송합니다, AI가 답변을 생성하는 중 오류가 발생했습니다. 좀더 이해하기 쉽게 질문하거나, 상담 관리자에게 문의해 주세요."

def generate_response_from_llama(user_message):
    try:
        if DEBUG_MODE:
            print(f"Generating response for user message: {user_message}")

        response = client.chat.completions.create(
            model="llama3.1-70b",
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": {user_message}}
            ],

            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_flight_times",
                        "description": "Get the flight times between two cities",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "departure": {
                                    "type": "string",
                                    "description": "The departure city (airport code)",
                                },
                                "arrival": {
                                    "type": "string",
                                    "description": "The arrival city (airport code)",
                                },
                            },
                            "required": ["departure", "arrival"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_antonyms",
                        "description": "Get the antonyms of any given words",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "word": {
                                    "type": "string",
                                    "description": "The word for which the opposite is required.",
                                },
                            },
                            "required": ["word"],
                        },
                    },
                },
            ],
        )

        if DEBUG_MODE:
            print(f"Llama Response 1 :", response)
        
        return response
    
    except Exception as e:
        if DEBUG_MODE:
            logging.error(f"Error while generating response from Llama: {str(e)}")
            print(f"Error while generating response from Llama: {str(e)}")
        return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다. 상담 관리자에게 문의해 주세요."

def generate_response_from_hcnllama(user_message):
    try:
        if DEBUG_MODE:
            print(f"Generating response for user message: {user_message}")

        # Ollama API를 사용하여 쿼리 재작성
        response = client.chat.completions.create(
            model="ms-phi3-4k",
            #model="myllama3",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2000,
            n=1,
            temperature=0.5,
        )

        if DEBUG_MODE:
            print(f"Response for user message: {response}")

        bot_reply = response.choices[0].message.content.strip()

        if DEBUG_MODE:
             print(f"Reply: {bot_reply}")

        return bot_reply
    
    except Exception as e:
        if DEBUG_MODE:
            logging.error(f"Error while generating response from Llama: {str(e)}")
            print(f"Error while generating response from Llama: {str(e)}")
        return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다. 상담 관리자에게 문의해 주세요."

def generate_response_from_claude(user_message):
    """Claude API를 사용하여 답변을 생성하는 함수"""
    try:
        if DEBUG_MODE:
            print(f"Generating response for user message: {user_message}")

        resopnse = client.messages.create(
            model="claude-3-5-sonnet-20240620",  # Claude model, you can choose the version
            max_tokens=1024,  # Set appropriate token limit
            tools=[
                {
                    "name": "get_delivery_date",
                    "description": "Get the delivery date in a given order ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The date of delivery",
                            }
                        },
                        "required": ["order_id"],
                    },
                },
                {
                    "name": "get_hcn_staffinfo",
                    "description": "Get the staff information in a staff ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The staff information",
                            }
                        },
                        "required": ["staff_id"],
                    },
                },
                {
                    "name": "get_hcn_staffnameinfo",
                    "description": "Get the staff information in a staff name",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The staff information",
                            }
                        },
                        "required": ["staff_name"],
                    },
                },
                {
                    "name": "get_hcn_counselinfo",
                    "description": "Get the counsel information in a given account ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The counsel information",
                            }
                        },
                        "required": ["account_id"],
                    },
                },
                {
                    "name": "get_hcn_billinfo",
                    "description": "Get the billing information in a account ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The billing information",
                            }
                        },
                        "required": ["account_id"],
                    },
                },
                {
                    "name": "get_hcn_productinfo",
                    "description": "Get the products information in a given account ID",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The products information",
                            }
                        },
                        "required": ["account_id"],
                    },
                }
            ],
            messages=[
                {   "role": "user", 
                    "content": user_message
                },
            ]
        )

        print("message.content = ", resopnse.content)

        if resopnse.stop_reason == "tool_use":
            function_name = resopnse.content[0].name
            function_text = resopnse.content[0].text
            function_id = resopnse.content[0].id

            if function_name == "get_delivery_date":
               order_id = resopnse.content[0].input
               resopnse_function = get_delivery_date(order_id)

            resopnse = client.messages.create(
                model="claude-3-5-sonnet-20240620",  # Claude model, you can choose the version
                max_tokens=1024,  # Set appropriate token limit
                tools=[
                    {
                        "name": "get_delivery_date",
                        "description": "Get the delivery date in a given order ID",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The date of delivery",
                                }
                            },
                            "required": ["order_id"],
                        },
                    },
                    {
                        "name": "get_hcn_staffinfo",
                        "description": "Get the staff information in a staff ID",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The staff information",
                                }
                            },
                            "required": ["staff_id"],
                        },
                    },
                    {
                        "name": "get_hcn_staffnameinfo",
                        "description": "Get the staff information in a staff name",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The staff information",
                                }
                            },
                            "required": ["staff_name"],
                        },
                    },
                    {
                        "name": "get_hcn_counselinfo",
                        "description": "Get the counsel information in a given account ID",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The counsel information",
                                }
                            },
                            "required": ["account_id"],
                        },
                    },
                    {
                        "name": "get_hcn_billinfo",
                        "description": "Get the billing information in a account ID",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The billing information",
                                }
                            },
                            "required": ["account_id"],
                        },
                    },
                    {
                        "name": "get_hcn_productinfo",
                        "description": "Get the products information in a given account ID",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The products information",
                                }
                            },
                            "required": ["account_id"],
                        },
                    }
                ],
                messages=[
                    {   "role": "user", 
                        "content": user_message
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": function_text
                            },
                            {
                                "type": "tool_use",
                                "id": function_id,
                                "name": "get_delivery_date",
                                "input": {"order_id"} 
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": function_id, # from the API response
                                "content": resopnse_function  # from running your tool
                            }
                        ]
                    }
                ]
            )

            print("message.content = ", resopnse.content)

        bot_reply = resopnse.content[0].text

        #print("bot_reply = ", bot_reply)

        if DEBUG_MODE:
             print(f"Generated Claude reply: {bot_reply}")
                
        return bot_reply
    
    except Exception as e:
        if DEBUG_MODE:
            logging.error(f"Error while generating response from Claude: {str(e)}")
            print(f"Error while generating response from Claude: {str(e)}")
        return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다. 상담 관리자에게 문의해 주세요."


def process_required_actions(run, thread):
    if DEBUG_MODE:
        print(f"process_required_actions HERE: {thread.id}")

    """필요한 작업을 처리하는 함수"""
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    tool_outputs = []
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Processing required actions for thread ID: {thread.id}")

    for tool_call in tool_calls:
        try:
            tool_output = process_tool_call(tool_call, tool_outputs)
        except Exception as e:
            if DEBUG_MODE:
                logging.error(f"Error processing tool call {tool_call.function.name}: {str(e)}")
                print(f"Error processing tool call {tool_call.function.name}: {str(e)}")  # Print for debugging

    # 모든 tool_output을 문자열로 변환하여 제출
    for tool_output in tool_outputs:
        if isinstance(tool_output['output'], dict):
            tool_output['output'] = json.dumps(tool_output['output'], ensure_ascii=False)
        elif not isinstance(tool_output['output'], str):
            tool_output['output'] = str(tool_output['output'])
    
    # Print for debugging
    # if DEBUG_MODE:
    #     print(f"process_required_actions tool_outputs: {tool_outputs}")

    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=tool_outputs
    )

    # if DEBUG_MODE:
    #     print(f"Tool outputs: {run}")

    # Run이 완료될 때까지 대기
    wait_on_run(run, thread)

def process_tool_call(tool_call, tool_outputs):
    """각 tool_call에 대해 처리하는 함수"""
    function_name = tool_call.function.name
    if DEBUG_MODE:
        logging.info("Processing tool call function: %s", function_name)
        print(f"Processing tool call: {function_name}")
    
    arguments = json.loads(tool_call.function.arguments)
    tool_call_id = tool_call.id

    # 예제 함수 이름에 따라 처리
    if function_name == "get_delivery_date":
        order_id = arguments.get('order_id')
        delivery_date = get_delivery_date(order_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": delivery_date
        })
    elif function_name == "get_hcn_accountinfo":
        account_id = arguments.get('symbol')
        account_info = get_hcn_accountinfo(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": account_info
        })
    elif function_name == "get_hcn_staffinfo":
        staff_id = arguments.get('staff_id')
        staff_info = get_hcn_staffinfo(staff_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": staff_info
        })
    elif function_name == "get_hcn_staffnameinfo":
        staff_name = arguments.get('staff_name')
        staff_info = get_hcn_staffnameinfo(staff_name)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": staff_info
        })
    elif function_name == "get_hcn_billinfo":
        account_id = arguments.get('symbol')
        bill_info = get_hcn_billinfo(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": bill_info
        })
    elif function_name == "get_hcn_productinfo":
        account_id = arguments.get('symbol')
        product_info = get_hcn_productinfo(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": product_info
        })
    elif function_name == "do_hcn_cancel_service":
        account_id = arguments.get('symbol')
        cancel_response = do_hcn_cancel_service(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": cancel_response
        })
    elif function_name == "do_hcn_as_request":
        account_id = arguments.get('symbol')
        response = do_hcn_as_request(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": response
        })
    elif function_name == "do_hcn_as_cancel":
        account_id = arguments.get('symbol')
        response = do_hcn_as_cancel(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": response
        })
    elif function_name == "get_hcn_account_monitoring":
        account_id = arguments.get('symbol')
        response = get_hcn_account_monitoring(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": response
        })
    elif function_name == "do_hcn_device_reset":
        account_id = arguments.get('symbol')
        response = do_hcn_device_reset(account_id)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": response
        }) 
    elif function_name == "do_trouble_to_send_sms":
        trouble_level = arguments.get('trouble_level')
        trouble_text = arguments.get('trouble_text')
        response = do_trouble_to_send_sms(trouble_level, trouble_text)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": response
        }) 
    elif function_name == "get_trouble_area":
        trouble_so = arguments.get('symbol')
        trouble_info = get_trouble_area(trouble_so)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": trouble_info
        }) 
    elif function_name == "generate_image":
        # 이미지 생성 함수 호출
        generate_img = arguments.get('symbol')
        image_url = generate_image(generate_img)
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": image_url
        }) 
    else:
        if DEBUG_MODE:
            logging.warning(f"Unrecognized function: {function_name}")
            print(f"Unrecognized function: {function_name}")  # Print for debugging
        tool_outputs.append({
            "tool_call_id": tool_call_id,
            "output": f"Function {function_name} not recognized."
        })

    # Print for debugging
    if DEBUG_MODE:
        print(f"Tool call {function_name} processed. Tool outputs: {tool_outputs}")

    # return tool_outputs

def wait_on_run(run, thread):
    """Run이 완료될 때까지 대기하는 함수"""
    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Run completed with status: {run.status}")
    
    return run

def submit_message(assistant_id, thread, user_message):
    """사용자 메시지를 스레드에 제출하는 함수"""
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=user_message
    )
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Submitted user message to thread: {thread.id}")
    
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Created run for thread: {thread.id}, run ID: {run.id}")
    
    return run


def get_final_response(thread):
    """스레드에서 최종 봇 응답만을 가져오는 함수"""
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc")

    if DEBUG_MODE:
        print(f"get_final_response messages: {messages}")

    if not messages:
        if DEBUG_MODE:
            logging.warning("No response received from thread.")
            print("No response received from thread.")  # Print for debugging
        return "No response received."

    bot_responses = []
    for message in messages:
        # 봇의 응답만 추가 (role이 'assistant'인 경우에만)
        if message.role == "assistant" :
            bot_responses.append(message.content[0].text.value)
            break

        if DEBUG_MODE:
            print(f"Messages: {message}")
    
    if DEBUG_MODE:
        print(f"Final response from messages: {messages}")

    
    # 봇의 모든 응답을 조합하여 최종 응답 생성
    final_response = "\n".join(bot_responses) if bot_responses else "No valid bot response received."
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Final response from bot: {final_response}")
    
    return final_response

def generate_image(user_message: str) -> str:
    """그림그려주고 저장하기"""
    response = openai.Image.create(
        prompt=user_message,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

def get_delivery_date(order_id: str) -> str:
    """더미 배송 날짜 조회 함수"""
    return "2024년 8월 22일"  # 예시 응답

def get_hcn_staffinfo(staff_id: str) -> dict:
    """직원 정보를 조회하는 함수"""
    db_path = os.getenv('DB_PATH', 'c:/openai-chatbot/hcn_staff.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if DEBUG_MODE:
        logging.info("Thread staff_id: %s", staff_id)
    cursor.execute('SELECT * FROM staff WHERE 사번 = ?', (staff_id,))
    row = cursor.fetchone()
    conn.close()

    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved staff info for ID {staff_id}: {row}")

    if row:
        return {
            "사번": row[0],
            "성함": row[1],
            "소속": row[2],
            "직급": row[3],
            "직책": row[4],
            "직무": row[5]
        }
    else:
        return {"error": "해당 사번의 직원을 찾을 수 없습니다"}
    
def get_hcn_staffnameinfo(staff_name: str) -> dict:
    """직원 이름으로 정보를 조회하는 함수"""
    db_path = os.getenv('DB_PATH', 'c:/openai-chatbot/hcn_staff.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    if DEBUG_MODE:
        logging.info("Thread staff_name: %s", staff_name)
    cursor.execute('SELECT * FROM staff WHERE 성함 = ?', (staff_name,))
    row = cursor.fetchone()
    conn.close()

    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved staff info for name {staff_name}: {row}")

    if row:
        return {
            "사번": row[0],
            "성함": row[1],
            "소속": row[2],
            "직급": row[3],
            "직책": row[4],
            "직무": row[5]
        }
    else:
        return {"error": "해당 이름의 직원을 찾을 수 없습니다"}

def get_hcn_accountinfo(account_id: str) -> str:
    """계정 정보를 조회하는 함수"""
    account_info = """[9월1일]HD실버3년약정하에 설치비 無/ 월:9,900원 청구/ 약정기간내 중도해지시 위약금발생 타구이사시 증빙서류 제출시 위약금 無//행사가 적용
                      [9월3일]신규가입자에게 쿠폰이 발행되어 사용하였지만 쿠폰 잔액이 남아 안내메세지를 보냄. 문자는 10월 6일에 보냈으며, 문자내용은 [HCN]VOD 쿠폰 남은잔액으로 VOD 시청가능! 10/8까지 (최신VOD-이웃사람)(담당자 : 마케팅기획팀 김보람, 내선번호 1518)
                      [9월4일]일부채널안나옴/리셋안내/해보고 다시 전화준다고 끊으심 
                      [9월5일]통화거부(도입부에서통화종료) 디지털TV 장기 이용고객 VOD 5천원 쿠폰발급/사용기간 11월30일까지"""
    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved account info for ID {account_id}: {account_info}")
    return account_info

def get_hcn_account_monitoring(account_id: str) -> str:
    """계정 정보를 조회하는 함수"""
    monitoring_info = f"고객번호 {account_id}_홍*동님의 방송STB와 인터넷 모뎀이 정상이 아닌것으로 보이며, 서비스 장애가 판단 됩니다. 리셋을 진행할까요?"

    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved account info for ID {account_id}: {monitoring_info}")
    return monitoring_info

def do_hcn_device_reset(account_id: str) -> str:
    """계정 정보를 조회하는 함수"""
    result_info = f"고객번호 {account_id}_홍*동님의 방송STB와 인터넷 모뎀이 리셋되었습니다."

    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved account info for ID {account_id}: {result_info}")
    return result_info

def get_hcn_billinfo(account_id: str) -> str:
    """청구서 정보를 조회하는 함수"""
    bill_info = f"고객번호 {account_id} 홍*동 고객님의 9월 현재까지 미납요금은 27,900원 입니다."
    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved bill info for ID {account_id}: {bill_info}")
    return bill_info

def get_hcn_productinfo(account_id: str) -> str:
    """상품 정보를 조회하는 함수"""
    product_info = f"고객번호 {account_id} 홍*동 고객님이 사용하시는 상품은 알뜰TV 단독 상품이며 요금은 월 13,200원 입니다."
    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved product info for ID {account_id}: {product_info}")
    return product_info

def do_hcn_cancel_service(account_id: str) -> str:
    """해지 접수 하는 함수"""
    cancel_response = f"고객번호 {account_id} 해지 접수 완료"
    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved cancel service info for ID {account_id}")
    return cancel_response

def do_hcn_as_request(account_id: str) -> str:
    """AS 접수 하는 함수"""
    request_response = f"고객번호 {account_id} AS 접수 완료"
    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved as request info for ID {account_id}")
    return request_response

def do_hcn_as_cancel(account_id: str) -> str:
    """AS 접수 취소 하는 함수"""
    cancel_response = f"고객번호 {account_id} AS 취소 완료"
    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved as request info for ID {account_id}")
    return cancel_response

def do_trouble_to_send_sms(trouble_level: str, trouble_text:str) -> str:
    """장애문자 전송"""
    if trouble_level == "1등급" :
        send_response = f"장애등급은 {trouble_level}으로 {trouble_text}으로 장애문자가 관련된 관리자와 담당자 195명에게 전송완료 되었습니다."
    elif trouble_level == "2등급" :     
        send_response = f"장애등급은 {trouble_level}으로 {trouble_text}으로 장애문자가 관련된 관리자와 담당자 98명에게 전송완료 되었습니다."
    elif trouble_level == "3등급" :        
        send_response = f"장애등급은 {trouble_level}으로 {trouble_text}으로 장애문자가 관련된 관리자와 담당자 39명에게 전송완료 되었습니다."
    else :
        send_response = f"장애등급 {trouble_level}으로 확인되지 않습니다. 장애 절차서 및 지침서를 확인하시기 바랍니다."    
    # Print for debugging
    if DEBUG_MODE:
        print(f"do_trouble_send_to_sms : {trouble_level}, {trouble_text}")
    return send_response

def get_trouble_area(trouble_so: str) -> str:
    """장애지역 알림"""
    if trouble_so == "금호방송":
        area_response = f"{trouble_so}의 장애는 2등급 장애로 지역은 고성동, 칠성동, 침산동, 산격동으로 파악됩니다. 장애문자를 관련 직원들에게 보낼까요?"
    elif trouble_so in ("충북방송" , "새로넷방송" , "경북방송", "부산방송"):
        area_response = f"{trouble_so}의 장애지역은 현재 없습니다."
    elif trouble_so in ("서초방송" , "동작방송" , "관악방송"):
        area_response = f"{trouble_so}은 장애 1등급으로 전지역 장애로 난리 났습니다. 장애문자를 전직원에게 보내야 겠지요?"        
    else :
        area_response = f"{trouble_so}은 HCN 서비스 지역이 아닙니다. 다시 확인해 주세요!!"  

    # Print for debugging
    if DEBUG_MODE:
        print(f"Retrieved as get_trouble_area {area_response}")
    return area_response
    
def get_weather_info(location):
    #url = api_url + "?q=" + location + "&key=" + weather_token  
     weather_info = {
        "location":"London",
        "temperature":8.0,
        "description":"Moderate rain"
     }
     return json.dumps(weather_info)

def log_conversation(user_message, bot_reply):
    """대화 내용을 파일에 기록하는 함수"""
    filename = get_filename()
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"User: {user_message}\n")
        f.write(f"Bot: {bot_reply}\n")
        f.write("\n")

    # Print for debugging
    if DEBUG_MODE:
        print(f"Logged conversation to {filename}")

def get_filename():
    """대화 내용을 저장할 파일 이름을 생성하는 함수"""
    now = datetime.now()
    folder_name = "chat_logs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = f"{folder_name}/comm_{now.strftime('%Y%m%d')}.txt"
    
    # Print for debugging
    if DEBUG_MODE:
        print(f"Generated log filename: {filename}")
    
    return filename


# STT 처리 부분 및 기타 유틸리티 함수들 추가

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
