import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. نظام التحليل اللغوي المتقدم
# ==========================================
def clean_text(text):
    if not text: return ""
    text = str(text).lower().strip()
    # توحيد الحروف العربية (إزالة الهمزات والتشكيل)
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # معالجة اللهجة العامية
    replacements = {"بدي": "اريد", "شو": "ما هو", "وين": "اين", "ليش": "لماذا", "قديش": "كم"}
    words = text.split()
    text = " ".join([replacements.get(w, w) for w in words])
    return text

# ==========================================
# 2. تحميل البيانات وتدريب "عقل" البوت
# ==========================================
def train_model(filename):
    if not os.path.exists(filename):
        return None, None, []
    try:
        df = pd.read_csv(filename, sep=',', encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = [c.lower() for c in df.columns]
        df.dropna(subset=['question', 'answer'], inplace=True)
        
        # تنظيف الأسئلة في الداتا
        df['clean_q'] = df['question'].apply(clean_text)
        
        # تحويل النصوص إلى مصفوفات رياضية (يفهم السياق)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # يفهم الكلمات المفردة والمزدوجة
        vectors = vectorizer.fit_transform(df['clean_q'])
        
        return vectorizer, vectors, df['answer'].tolist()
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None, []

print("⚙️ جاري تشغيل محرك Nashama AI...")
vec_ar, mtx_ar, ans_ar = train_model("jordan_data_ar.csv")
vec_en, mtx_en, ans_en = train_model("jordan_data_en.csv")

# ==========================================
# 3. وظيفة البحث في الإنترنت
# ==========================================
def search_web(query, lang="ar"):
    try:
        print(f"🌍 جاري البحث في الإنترنت عن: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            if results:
                res = results[0]
                header = "لم أجد ذلك في ملفاتي، ولكن إليك ما وجدته في النت:\n" if lang=="ar" else "Found online:\n"
                return f"{header}{res['body']}\n🔗 المصدر: {res['title']} ({res['href']})"
    except:
        return None
    return None

# ==========================================
# 4. معالجة الطلبات (API)
# ==========================================
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        user_q = data.get("question", "").strip()
        if not user_q: return jsonify({"answer": "لم تكتب شيئاً!"})

        # تحديد اللغة
        is_en = bool(re.search(r'[a-zA-Z]', user_q))
        lang = "en" if is_en else "ar"
        
        # اختيار النموذج المناسب
        v, m, a = (vec_en, mtx_en, ans_en) if is_en else (vec_ar, mtx_ar, ans_ar)
        
        # 1. البحث في الداتا المحلية
        cleaned_q = clean_text(user_q)
        user_vec = v.transform([cleaned_q])
        similarity = cosine_similarity(user_vec, m)
        best_idx = similarity.argmax()
        score = similarity[0][best_idx]

        print(f"🔍 السؤال: {user_q} | الثقة: {int(score*100)}%")

        # 2. اتخاذ القرار
        if score >= 0.45: # إذا نسبة التشابه قوية (45% فأكثر)
            return jsonify({"answer": a[best_idx], "source": "database"})
        else:
            # 3. إذا لم يجد، يبحث في النت
            web_result = search_web(user_q, lang)
            if web_result:
                return jsonify({"answer": web_result, "source": "internet"})
            else:
                fallback = "عذراً، لم أجد إجابة دقيقة." if lang=="ar" else "Sorry, I couldn't find an answer."
                return jsonify({"answer": fallback, "source": "none"})

    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)