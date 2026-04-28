const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// --- 1. دالة ذكية لتحويل الروابط النصية إلى روابط قابلة للضغط ---
function linkify(text) {
    const urlPattern = /(\b(https?):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig;
    return text.replace(urlPattern, function(url) {
        return `<a href="${url}" target="_blank" class="chat-link">${url}</a>`;
    });
}

async function sendMessage() {
   const data = await res.json();
            removeTyping(typingId);

            // --- تحديث واجهة الانحدار الخطي ---
            const statsDiv = document.getElementById('regression-stats');
            const barFill = document.getElementById('conf-bar-fill');
            const percText = document.getElementById('conf-perc');

            if (data.confidence) {
                statsDiv.style.display = 'flex';
                barFill.style.width = data.confidence + '%';
                percText.innerText = data.confidence + '%';
                
                // تغيير اللون حسب الثقة
                if(data.confidence > 75) barFill.style.background = '#007a3d'; // أخضر
                else if(data.confidence > 50) barFill.style.background = '#ffc107'; // أصفر
                else barFill.style.background = '#ce1126'; // أحمر
            } else {
                statsDiv.style.display = 'none'; // إخفاء الشريط إذا كان الرد من Gemini
            }

            let formatted = linkify(data.answer).replace(/\n/g, '<br>');
            appendMsg(formatted, 'bot', true);
    try {
        const response = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            mode: 'cors',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ 
                question: question,
                session_id: "user_session_001" 
            })
        });

        if (!response.ok) throw new Error('خطأ في استجابة السيرفر');

        const data = await response.json();
        
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) loadingElement.remove();

        // إرسال الإجابة للدالة التي تدعم الروابط
        appendMessage(data.answer, 'bot-msg');

    } catch (error) {
        const loadingElement = document.getElementById(loadingId);
        if (loadingElement) loadingElement.remove();
        appendMessage("تعذر الاتصال بالسيرفر. تأكد أن شاشة البايثون السوداء في VS Code ما زالت تعمل.", 'bot-msg');
        console.error("Fetch Error:", error);
    }
}

// --- 2. تعديل دالة إضافة الرسائل لدعم HTML (الروابط) ---
function appendMessage(text, className, id = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    if (id) msgDiv.id = id;

    // إذا كانت الرسالة من البوت، نقوم بتفعيل الروابط فيها
    if (className === 'bot-msg') {
        msgDiv.innerHTML = linkify(text); 
    } else {
        msgDiv.textContent = text; // رسائل المستخدم تبقى نصاً عادياً للأمان
    }

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});