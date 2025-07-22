import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import joblib
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

# بارگذاری مدل‌ها
svm_model = joblib.load('svm_model.pkl')
bert_pipeline = pipeline("sentiment-analysis")

def analyze_with_svm(text):
    return svm_model.predict([text])[0]

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    bert_result = bert_pipeline(text)[0]
    label_bert = bert_result['label']
    label_svm = analyze_with_svm(text)
    response = f"""📊 تحلیل احساس:
🔹 BERT: {label_bert}
🔸 SVM: {label_svm}
    """
    await update.message.reply_text(response)

async def send_comparison(update: Update, context: ContextTypes.DEFAULT_TYPE):
    metrics = ['Accuracy', 'Training Time', 'Inference Time']
    bert_values = [0.91, 300, 15]
    svm_values = [0.82, 25, 3]
    bert_norm = [b / max(b, s) for b, s in zip(bert_values, svm_values)]
    svm_norm = [s / max(b, s) for b, s in zip(bert_values, svm_values)]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, bert_norm, width, label='BERT', color='skyblue')
    ax.bar(x + width/2, svm_norm, width, label='SVM', color='lightgreen')
    ax.set_title("Comparison of BERT and SVM")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Normalized Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.close()
    await update.message.reply_photo(photo=open("comparison.png", 'rb'))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! یک متن بفرست تا با مدل BERT و SVM تحلیلش کنم.\nدستور /compare هم برای دیدن نمودار عملکرد است.")

# اجرای برنامه
app = ApplicationBuilder().token(os.getenv("8092558246:AAF2l5rmwiFp9TEgQKAPVA95BXx7q6zdItc")).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("compare", send_comparison))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.run_polling()

