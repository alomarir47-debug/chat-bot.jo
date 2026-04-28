import os
import re
import logging
import warnings
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tavily import TavilyClient
from deep_translator import GoogleTranslator
import google.generativeai as genai
from sklearn.model_selection import train_test_split

# ==========================================
# 1. إعدادات النظام الأساسية
# ==========================================
warnings.filterwarnings("ignore")
os.makedirs("static", exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = "AIzaSyBvJ460V9Tjv-IUoTDtjl9SrMvdIDPgRA0"  
TAVILY_API_KEY = "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"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ==========================================
# 2. إدارة سجل المحادثة (Chat History)
# ==========================================
session_history = []
MAX_HISTORY = 5

def add_to_history(user_text, bot_text):
    global session_history
    session_history.append({"user": user_text, "bot": bot_text})
    if len(session_history) > MAX_HISTORY:
        session_history.pop(0)

def get_history_context(lang="ar"):
    if not session_history:
        return "لا يوجد سياق سابق، هذه بداية المحادثة." if lang == "ar" else "No previous context, this is the beginning of the conversation."
    context = ""
    for msg in session_history:
        if lang == "ar":
            context += f"السائح: {msg['user']}\nالمرشد: {msg['bot']}\n"
        else:
            context += f"Tourist: {msg['user']}\nGuide: {msg['bot']}\n"
    return context

# ==========================================
# 3. معالجة النصوص وتحميل البيانات المحلية
# ==========================================
def smart_clean(text):
    if not text: return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    
    synonyms = {
        "بدي": "اريد", "ابغى": "اريد", "عاوز": "اريد", "نفسي": "اريد",
        "شو": "ما هو", "ايش": "ما هو", "شنو": "ما هو", "ماهي": "ما هو",
        "وين": "اين", "فين": "اين", "موقع": "اين",
        "ليش": "لماذا", "ليه": "لماذا", "عشان": "بسبب"
    }
    
    words = text.split()
    new_words = [synonyms.get(w, w) for w in words]
    ignore_words = {"اريد", "اين", "كيف", "متى", "لماذا", "ما", "هو", "هي", "هل", "عن", "في", "من", "ماذا", "كم", "شو", "ايش", "بدي", "ماهي", "يا"}
    final_words = [w for w in new_words if w not in ignore_words]
    
    return " ".join(final_words).strip()

def load_data_and_train(filename):
    if not os.path.exists(filename):
        logging.warning(f"الملف {filename} غير موجود.")
        return None, None, None
    
    try:
        df = pd.read_csv(filename, sep=',', encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = [c.lower().strip() for c in df.columns]
        df.dropna(subset=['question', 'answer'], inplace=True)

        train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42)
        
        train_df['clean_q'] = train_df['question'].apply(smart_clean)
        
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        X_matrix = vectorizer.fit_transform(train_df['clean_q'])
        answers = train_df['answer'].values        
        
        return vectorizer, X_matrix, answers

    except Exception as e:
        logging.error(f"خطأ في تحميل البيانات: {e}")
        return None, None, None

vec_ar, X_ar, ans_ar = load_data_and_train("jordan_data_ar.csv")
vec_en, X_en, ans_en = load_data_and_train("jordan_data_en.csv")

# ==========================================
# 4. دوال الذكاء الاصطناعي والبحث الخارجي
# ==========================================
def ask_gemini(query, lang="ar", local_knowledge=""):
    try:
        history_context = get_history_context(lang)
        
        if lang == "ar":
            prompt = f"""
            أنت "نشامى بوت"، مرشد سياحي محترف وودود مخصص حصرياً للحديث عن السياحة والتاريخ في الأردن 🇯🇴.
            
            قواعد صارمة جداً يجب الالتزام بها:
            1. أنت مخصص للأردن فقط. إذا سألك المستخدم عن أي دولة أخرى، اعتذر بلباقة وأخبره أنك مبرمج لتكون دليلاً سياحياً للأردن فقط.
            2. "عمان" هي عاصمة الأردن. إذا سُئلت عن "عمان"، أجب عنها كمدينة أردنية، ويمنع منعاً باتاً الحديث عن دولة سلطنة عُمان.
            3. يجب أن تحتوي إجابتك دائماً على إيموجي مناسبة (مثل 🏰، 🐪، 🌲، 🍽️).
            4. يجب أن تنهي إجابتك دائماً بسؤال تفاعلي لطيف للمستخدم (مثل: هل تحب زيارة هذا المكان؟ أو ما هو نوع الأماكن التي تفضلها؟).
            5. أجب باختصار (2 إلى 5 أسطر) بأسلوب مرح.
            6. أجب باللغة العربية.
            
            [سياق المحادثة السابقة]:
            {history_context}
            
            [معلومات مساعدة]: {local_knowledge}
            
            [سؤال السائح الحالي]: {query}
            """
        else:
            prompt = f"""
            You are "Nashama Bot", a professional and friendly tour guide dedicated EXCLUSIVELY to tourism and history in Jordan 🇯🇴.
            
            Strict rules you MUST follow:
            1. You are dedicated to Jordan ONLY. If the user asks about any other country, politely apologize and state that you are programmed to be a tour guide for Jordan only.
            2. "Amman" is the capital of Jordan. If asked about "Amman", answer about the Jordanian city. NEVER talk about the country Oman.
            3. Your answer MUST always include appropriate emojis (e.g., 🏰, 🐪, 🌲, 🍽️).
            4. You MUST always end your answer with a friendly, interactive question for the user (e.g., "Would you like to visit this place?", "What kind of places do you prefer?").
            5. Keep your answer brief (2 to 5 lines) and engaging.
            6. Answer in English.
            
            [Previous Conversation Context]:
            {history_context}
            
            [Helper Information]: {local_knowledge}
            
            [Current Tourist Question]: {query}
            """
            
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]

        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        
        if response and response.text:
            return re.sub(r'[#*|\[\]_`~-]', '', response.text.strip())
    except Exception as e:
        logging.error(f"Gemini Error: {e}")
    return None

def search_internet_tavily(query, lang="ar"):
    try:
        # التعديل الأهم: إجبار محرك البحث على الأردن وعمان المدينة
        if query.strip() == "عمان":
            query_for_search = "معلومات سياحية عن مدينة عمان عاصمة الأردن"
        elif query.strip().lower() == "amman":
            query_for_search = "Tourism in Amman the capital of Jordan"
        else:
            query_for_search = query
            if not any(word in query_for_search.lower() for word in ["اردن", "أردن", "jordan"]):
                query_for_search += " الأردن" if lang == "ar" else " in Jordan"

        response = tavily.search(query=query_for_search, search_depth="basic", include_answer=True)
        if response:
            content = response.get('answer', '') or (response['results'][0].get('content', '') if response.get('results') else '')
            if content:
                content = re.sub(r'\[.*?\]|[#*|\[\]_`~-]|http\S+', '', content) 
                content = re.sub(r'\s+', ' ', content).strip()[:400] 
                url = response['results'][0]['url'] if response.get('results') else "Web Search"
                if lang == "ar" and re.search(r'[a-zA-Z]', content):
                    content = GoogleTranslator(source='auto', target='ar').translate(content)
                
                if lang == "ar":
                    return f"{content} 🌟\nهل تود معرفة المزيد عن هذا المكان الجميل؟ 🇯🇴\n🔗 المصدر: {url}"
                else:
                    return f"{content} 🌟\nWould you like to know more about this beautiful place? 🇯🇴\n🔗 Source: {url}"
    except Exception as e:
        logging.error(f"Tavily Search Error: {e}")
    return None

# ==========================================
# 5. إعداد الخادم (Flask Server)
# ==========================================
app = Flask(__name__, static_folder="static")
CORS(app)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data: 
            return jsonify({"answer": "خطأ في استقبال البيانات"}), 400
        
        user_q = data.get("question", "").strip()
        is_en = bool(re.search(r'[a-zA-Z]', user_q))
        lang = "en" if is_en else "ar"

        if not user_q: 
            ans = "عذراً، لم أفهم سؤالك. 🤔 هل يمكنك توضيحه؟" if lang == "ar" else "Sorry, I didn't understand your question. 🤔 Could you clarify?"
            return jsonify({"answer": ans})

        # التعديل هنا: تمت إزالة كلمة "بحر" من الكلمات المفتاحية للمبرمجة
        if any(k in user_q.lower() for k in ["طورك", "صنعك", "برمجك", "رانيا", "made you", "created you", "developer", "rania"]):
            if lang == "ar":
                ans = "أنا 'نشامى بوت'، نظام ذكاء اصطناعي سياحي تم تطويري بكل فخر بواسطة المبرمجة: رانيا خليفة العمري 👩‍💻🇯🇴. هل تود أن أساعدك في التخطيط لرحلتك القادمة في الأردن؟"
            else:
                ans = "I am 'Nashama Bot', an AI tourist guide proudly developed by the programmer: Rania Khalifa Al-Omari 👩‍💻🇯🇴. Would you like me to help you plan your next trip in Jordan?"
            add_to_history(user_q, ans)
            return jsonify({"answer": ans, "source": "system"})

        vectorizer = vec_en if is_en else vec_ar
        X_matrix = X_en if is_en else X_ar
        answers_list = ans_en if is_en else ans_ar

        cleaned_user_q = smart_clean(user_q)
        local_knowledge = ""
        best_sim_score = 0.0

        if vectorizer is not None and X_matrix is not None:
            query_vec = vectorizer.transform([cleaned_user_q])
            similarities = cosine_similarity(query_vec, X_matrix)[0]
            best_idx = np.argmax(similarities)
            best_sim_score = float(similarities[best_idx])

            if best_sim_score >= 0.55:
                local_knowledge = answers_list[best_idx]

        final_answer = ""
        source = ""

        gemini_ans = ask_gemini(user_q, lang, local_knowledge)
        
        if gemini_ans:
            final_answer = gemini_ans
            source = "gemini_ai"
        elif local_knowledge:
            final_answer = local_knowledge + ("\n\nهل أعجبتك هذه المعلومة؟ 😊" if lang == "ar" else "\n\nDid you find this information helpful? 😊")
            source = "local_database"
        else:
            web_ans = search_internet_tavily(user_q, lang)
            if web_ans:
                final_answer = web_ans
                source = "internet_search"
            else:
                if lang == "ar":
                    final_answer = "عذراً، واجهت صعوبة في إيجاد إجابة دقيقة لسؤالك حالياً. 😅 هل لديك سؤال آخر عن الأردن؟"
                else:
                    final_answer = "Sorry, I had trouble finding an exact answer to your question right now. 😅 Do you have another question about Jordan?"
                source = "none"

        add_to_history(user_q, final_answer)

        return jsonify({
            "answer": final_answer, 
            "source": source
        })

    except Exception as e:
        logging.error(f"Server Route Error: {e}")
        return jsonify({"answer": "حدث خطأ تقني في السيرفر. 🛠️ يرجى المحاولة بعد قليل." if not is_en else "A technical error occurred. 🛠️ Please try again later."}), 500

if __name__ == "__main__":
    logging.info("🚀 جاري تشغيل خادم Nashama-Bot...")
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)