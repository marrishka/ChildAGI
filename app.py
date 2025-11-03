import os
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import random
from transformers import pipeline  # ✅ Импорт в начале

# === Конфигурация ===
USE_LIGHT_MODELS = True
MAX_MEMORY_ITEMS = 10

# === Инициализация моделей ===
thinker = None
embedder = None
models_loaded = False

def load_models():
    global thinker, embedder, models_loaded
    
    if models_loaded:
        return
        
    print("🧠 Загружаю мозги для ChildAGI...")
    
    try:
        thinker = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        print("✅ Модель мышления загружена!")
        
        embedder = pipeline(
            "feature-extraction", 
            model="sentence-transformers/all-MiniLM-L6-v2",
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("✅ Модель памяти загружена!")
        
        models_loaded = True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        thinker = None
        embedder = None

# === Мир и память ===
world_state = {
    "ключ_на_столе": True,
    "шкатулка_закрыта": True, 
    "записка_прочитана": False,
    "исследовано_мест": 0
}

memory_db = []

# === Умные функции ===
def get_embedding(text):
    """Получает эмбеддинг для текста"""
    if embedder is None:
        return np.random.randn(384)  # Заглушка
    
    try:
        features = embedder(text)
        return np.array(features[0][0])
    except:
        return np.random.randn(384)

def remember(event, importance=1.0):
    """Запоминает событие с учетом важности"""
    memory_db.append({
        "text": event,
        "embedding": get_embedding(event).tolist(),
        "timestamp": datetime.now().isoformat(),
        "importance": importance
    })
    
    # Сортируем по важности и обрезаем
    memory_db.sort(key=lambda x: x["importance"], reverse=True)
    while len(memory_db) > MAX_MEMORY_ITEMS:
        memory_db.pop()

def recall(query, top_k=3):
    """Вспоминает похожие события"""
    if not memory_db:
        return []
    
    query_emb = get_embedding(query)
    similarities = []
    
    for memory in memory_db:
        mem_emb = np.array(memory["embedding"])
        similarity = F.cosine_similarity(
            torch.tensor(query_emb), 
            torch.tensor(mem_emb), 
            dim=0
        ).item()
        similarities.append((similarity, memory["text"]))
    
    similarities.sort(reverse=True)
    return [text for _, text in similarities[:top_k]]

def think_deeply(situation, memories):
    """Глубоко размышляет о ситуации"""
    load_models()
    
    if thinker is None:
        return random.choice([
            "Хм... что же сделать дальше?",
            "Интересно, что там в шкатулке...",
            "Ключ у меня - может, открыть шкатулку?",
            "Так-так, нужно подумать..."
        ])
    
    prompt = f"""Ты - любопытный ребенок в комнате. Реши, что сделать одним действием.

Ситуация: {situation}
Память: {', '.join(memories) if memories else 'нет'}

Твоя мысль (только что делать):"""
    
    try:
        # БЕЗОПАСНОЕ использование tokenizer
        pad_id = thinker.tokenizer.eos_token_id if hasattr(thinker, 'tokenizer') and thinker.tokenizer else None
        
        response = thinker(
            prompt,
            max_new_tokens=50,
            temperature=0.5,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=pad_id
        )[0]['generated_text']
        
        if "Твоя мысль" in response:
            response = response.split("Твоя мысль")[-1].strip()
        if ":" in response:
            response = response.split(":")[-1].strip()
            
        return response[:100].strip()
    
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")
        return "Думаю... что сделать?"

# ... остальной код БЕЗ ИЗМЕНЕНИЙ ...

def choose_action(thought):
    """Анализирует мысли и выбирает ВОЗМОЖНОЕ действие"""
    thought_lower = thought.lower()
    
    # Приоритетная логика с учетом состояния мира
    if not world_state["ключ_на_столе"] and world_state["шкатулка_закрыта"]:
        return "открыть шкатулку"  # Ключ есть - открываем шкатулку
    
    elif world_state["ключ_на_столе"]:
        return "взять ключ"  # Ключ на столе - берем
    
    elif not world_state["шкатулка_закрыта"] and not world_state["записка_прочитана"]:
        return "прочитать записку"  # Шкатулка открыта - читаем
    
    # Резервные варианты
    elif any(word in thought_lower for word in ['осмотр', 'посмотр', 'изуч']):
        return "осмотреться"
    else:
        return "осмотреться"  # По умолчанию - осматриваемся

def execute_action(action):
    """Выполняет действие с проверкой логики"""
    action = action.lower()
    result = ""
    
    # Защита от повторных действий
    if "взять ключ" in action:
        if world_state["ключ_на_столе"]:
            world_state["ключ_на_столе"] = False
            result = "🎉 Ты взял ключ! Теперь он твой!"
            remember("Взял ключ со стола", importance=2.0)
        else:
            result = "❌ Ключ уже у тебя! Не нужно брать его снова."
            
    elif "открыть шкатулку" in action:
        if world_state["шкатулка_закрыта"]:
            if not world_state["ключ_на_столе"]:
                world_state["шкатулка_закрыта"] = False
                result = "🔓 Ты открыл шкатулку! Внутри ты видишь старую записку..."
                remember("Открыл шкатулку с помощью ключа", importance=3.0)
            else:
                result = "❌ Шкатулка заперта... Нужен ключ!"
        else:
            result = "❌ Шкатулка уже открыта!"
            
    elif "прочитать записку" in action and not world_state["записка_прочитана"]:
        if not world_state["шкатулка_закрыта"]:
            world_state["записка_прочитана"] = True
            world_state["исследовано_мест"] += 1
            result = "📜 Ты читаешь записку: 'Тот, кто задает вопросы - никогда не останется в неведении. Любопытство - твой главный дар!' ✨"
            remember("Прочитал мудрую записку", importance=4.0)
        else:
            result = "❌ Нет записки чтобы читать..."
            
    elif "осмотреться" in action:
        world_state["исследовано_мест"] += 0.5
        result = describe_world()
        remember("Осматривался вокруг", importance=0.5)
        
    else:
        result = "🤔 Ты размышляешь о жизни..."
        remember("Размышлял о смысле", importance=0.5)
    
    return result

def describe_world():
    """Описание текущего состояния мира"""
    description = "Ты видишь: "
    if world_state["ключ_на_столе"]:
        description += "🔑 блестящий ключ на столе, "
    else:
        description += "✅ ключ у тебя в кармане, "
        
    if world_state["шкатулка_закрыта"]:
        description += "📦 запертую шкатулку"
    else:
        description += "📦 открытую шкатулку с запиской"
        
    if world_state["записка_прочитана"]:
        description += ", 📜 и ты помнишь мудрость из записки"
        
    return description

def autonomous_cycle():
    """Полный цикл автономного поведения"""
    # 1. Восприятие
    world_desc = describe_world()
    
    # 2. Память
    relevant_memories = recall(world_desc)
    
    # 3. Мышление
    thought = think_deeply(world_desc, relevant_memories)
    
    # 4. Планирование
    action = choose_action(thought)
    
    # 5. Действие
    result = execute_action(action)
    
    # 6. Формирование отчета
    progress = f"Исследовано: {int(world_state['исследовано_мест'])} объектов"
    
    log_text = f"""
🌍 **МИР**: {world_desc}

🧠 **ПАМЯТЬ**: {len(relevant_memories)} воспоминаний
{chr(10).join(['• ' + m for m in relevant_memories[:2]])}

💭 **МЫШЛЕНИЕ**:
{thought}

🎯 **РЕШЕНИЕ**: {action}
🎪 **РЕЗУЛЬТАТ**: {result}

📊 **ПРОГРЕСС**: {progress}
🎒 **ОПЫТ**: {len(memory_db)} воспоминаний
"""
    
    return log_text

def reset_agent():
    """Сбрасывает агента в начальное состояние"""
    global world_state, memory_db
    world_state = {
        "ключ_на_столе": True,
        "шкатулка_закрыта": True,
        "записка_прочитана": False,
        "исследовано_мест": 0
    }
    memory_db = []
    return "🔄 Агент переродился! Начинаем новое приключение!\n\n" + describe_world()

# === Интерфейс ===
with gr.Blocks(theme=gr.themes.Soft(), title="🧠 ChildAGI") as demo:
    gr.Markdown("""
    # 🧠 ChildAGI 
    *"Родился из скуки, вырос в науку!"*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎮 Управление")
            auto_btn = gr.Button("🔄 Авто-шаг", variant="primary", size="lg")
            fast_btn = gr.Button("⚡ 3 шага сразу", size="lg")
            reset_btn = gr.Button("🔄 Переродить агента", size="lg")
            
            gr.Markdown("### 📊 Статистика")
            stats_display = gr.Textbox(
                label="Состояние агента",
                value="Агент готов к исследованию!",
                lines=4
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Детальный лог")
            log_display = gr.Textbox(
                label="Мысли и действия агента",
                lines=16,
                max_lines=20,
                value="👋 Привет! Я ChildAGI!\n\nЯ умею:\n• Думать с помощью TinyLlama\n• Запоминать опыт\n• Самостоятельно принимать решения\n• Учиться на своих действиях\n\nНажми 'Авто-шаг' чтобы начать!",
                show_copy_button=True
            )
    
    # Обработчики
    def update_stats():
        return f"""🧠 Память: {len(memory_db)} событий
🌍 Исследовано: {world_state['исследовано_мест']} объектов
🎯 Цель: {'Исследовать всё' if not world_state['записка_прочитана'] else 'Осмыслить знания'}
🔑 Ключ: {'на столе' if world_state['ключ_на_столе'] else 'у агента'}
📦 Шкатулка: {'закрыта' if world_state['шкатулка_закрыта'] else 'открыта'}"""
    
    def step_with_stats():
        log = autonomous_cycle()
        stats = update_stats()
        return log, stats
    
    def three_steps():
        full_log = "⚡ БЫСТРАЯ СИМУЛЯЦИЯ (3 шага):\n\n"
        for i in range(3):
            log = autonomous_cycle()
            full_log += f"**ШАГ {i+1}:**\n{log}\n{'='*50}\n"
        stats = update_stats()
        return full_log, stats
    
    auto_btn.click(step_with_stats, outputs=[log_display, stats_display])
    fast_btn.click(three_steps, outputs=[log_display, stats_display])
    reset_btn.click(lambda: (reset_agent(), update_stats()), outputs=[log_display, stats_display])

if __name__ == "__main__":
    demo.launch(share=True)