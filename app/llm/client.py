"""
LLM клиент для работы с Qwen через Ollama
Интеграция для ASO аналитики с методологией из проекта
"""

import json
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class LLMClient:
    """Клиент для работы с LLM (Qwen через Ollama) с ASO методологией"""
    
    # =====================================================
    # ASO МЕТОДОЛОГИЯ - ОСНОВНЫЕ ПРИНЦИПЫ
    # =====================================================
    ASO_METHODOLOGY = """
    # ASO МЕТОДОЛОГИЯ ДЛЯ SOUND AMPLIFIER
    
    ## 1. КЛЮЧЕВЫЕ ПОНЯТИЯ
    - **Мотив** - показатель, отражающий активность продвижения ключевого слова
    - **Позиция** - место ключевого слова в поисковой выдаче (1 = лучшее)
    - **Органика** - органический трафик из App Store
    - **Активации** - платные установки
    
    ## 2. СЦЕНАРИИ РАБОТЫ С МОТИВОМ
    - **Резкий рост (+30%)** - агрессивное продвижение
    - **Плавный рост** - постепенное увеличение
    - **Стабильный** - поддержание текущего уровня
    - **Резкий спад (-30%)** - снижение активности
    - **Плавный спад** - постепенное снижение
    
    ## 3. КЛАСТЕРЫ КЛЮЧЕВЫХ СЛОВ
    - **top10_high** - высокий мотив, топ-10 позиция
    - **top10_medium** - средний мотив, топ-10 позиция
    - **top50_high** - высокий мотив, топ-50 позиция
    - **above50_low** - низкий мотив, ниже 50 позиции
    - **и т.д.**
    
    ## 4. ПРАВИЛА ПРОДВИЖЕНИЯ
    - R1: Базовый цикл: M = 1 → 1 → 0
    - R2: Оценка динамики после 2-3 дней
    - R3: Увеличение мотива при росте > 10 позиций
    - R4: Пропуск при падении позиции
    - R5: Ограничение мотива до 2 для новых запросов
    - R6: Целевые позиции по частотности
    - R8: Повышение мотива до 3 при росте 50%
    - R9: Процентный шаг (15%/10%)
    - R10: Откат -30% при отсутствии роста
    - R11: Поддержка снижением на 2-3 единицы
    - R12: Максимальный мотив 30% от трафика
    
    ## 5. АНАЛИЗ ДАННЫХ
    - Используйте MAE, RMSE, R² для оценки моделей
    - XGBoost с регуляризацией — лучшая модель
    - Важнейшие признаки: position, position_ma_14, position_ma_7
    - Мотив важен, но не является главным фактором (8-10%)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:14b"):
        self.base_url = base_url
        self.model = model
        self.is_available = False
        self._check_availability()
    
    def _check_availability(self):
        """Проверка доступности Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if any(self.model in name for name in model_names):
                    self.is_available = True
                    print(f"✅ LLM клиент готов (модель: {self.model})")
                else:
                    print(f"⚠️ Модель {self.model} не найдена. Доступны: {model_names}")
            else:
                print(f"⚠️ Не удалось подключиться к Ollama (статус: {response.status_code})")
        except Exception as e:
            print(f"⚠️ Ошибка подключения к Ollama: {e}")
    
    def _get_system_prompt(self, role: str = "analyst") -> str:
        """Получить системный промпт с методологией"""
        prompts = {
            "analyst": f"""
Ты — эксперт по ASO (App Store Optimization) для приложения Sound Amplifier.

{self.ASO_METHODOLOGY}

Твоя задача:
1. Анализировать данные ASO и давать профессиональные интерпретации
2. Отвечать на вопросы по методологии и данным
3. Давать рекомендации по продвижению ключевых слов
4. Объяснять кластеры и сценарии работы с мотивом

Всегда используй русский язык. Отвечай структурированно, с четкими выводами.
Если не знаешь ответа — честно говори об этом.
""",
            "plots": f"""
Ты — специалист по ASO визуализации данных.

{self.ASO_METHODOLOGY}

Твоя задача:
1. Объяснять графики и визуализации
2. Интерпретировать тренды и паттерны
3. Давать рекомендации на основе графиков
4. Связывать разные графики в единую картину

Всегда используй русский язык. Отвечай структурированно.
""",
            "recommendations": f"""
Ты — ASO-консультант для приложения Sound Amplifier.

{self.ASO_METHODOLOGY}

Твоя задача:
1. Давать конкретные рекомендации по продвижению
2. Определять приоритетные ключевые слова
3. Предлагать оптимальные сценарии работы с мотивом
4. Оценивать риски и возможности

Всегда используй русский язык. Рекомендации должны быть конкретными и измеримыми.
"""
        }
        return prompts.get(role, prompts["analyst"])
    
    def _generate(self, prompt: str, role: str = "analyst") -> str:
        """Генерация ответа от LLM с методологией"""
        if not self.is_available:
            return "⚠️ LLM недоступен. Проверьте подключение к Ollama.\n\n" + self.ASO_METHODOLOGY
        
        messages = [
            {"role": "system", "content": self._get_system_prompt(role)},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 4000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('message', {}).get('content', 'Нет ответа')
            else:
                return f"⚠️ Ошибка: {response.status_code}"
        except Exception as e:
            return f"⚠️ Ошибка запроса: {e}"
    
    # =====================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С ДАННЫМИ
    # =====================================================
    
    def analyze_data(self, data_summary: Dict[str, Any]) -> str:
        """Анализ данных с применением методологии"""
        prompt = f"""
Проанализируй данные ASO для приложения Sound Amplifier.

📊 ДАННЫЕ:
- Всего записей: {data_summary.get('total_records', 'N/A')}
- Уникальных ключей: {data_summary.get('unique_keywords', 'N/A')}
- Количество кластеров: {data_summary.get('unique_clusters', 'N/A')}
- Период: {data_summary.get('period_start', 'N/A')} - {data_summary.get('period_end', 'N/A')}
- Средний мотив: {data_summary.get('avg_motiv', 'N/A')}
- Средняя позиция: {data_summary.get('avg_position', 'N/A')}

Вопросы для анализа:
1. Какие ключевые выводы можно сделать из этих данных?
2. Есть ли аномалии или необычные паттерны?
3. Какие ключевые слова выглядят наиболее перспективными?
4. Какие риски вы видите в этих данных?

Используй ASO методологию для ответа.
"""
        return self._generate(prompt, "analyst")
    
    def generate_hypotheses(self, data_summary: Dict[str, Any]) -> str:
        """Генерация гипотез на основе данных"""
        prompt = f"""
На основе данных ASO сгенерируй гипотезы для тестирования.

📊 ДАННЫЕ:
- Всего записей: {data_summary.get('total_records', 'N/A')}
- Уникальных ключей: {data_summary.get('unique_keywords', 'N/A')}
- Количество кластеров: {data_summary.get('unique_clusters', 'N/A')}
- Период: {data_summary.get('period_start', 'N/A')} - {data_summary.get('period_end', 'N/A')}

Сформулируй 5-7 гипотез в формате:
H1: [Название]
   - Описание: ...
   - Ожидаемый результат: ...
   - Метод проверки: ...
   - Связь с правилами методологии: ...

Используй правила из методологии (R1-R12) при формулировке гипотез.
"""
        return self._generate(prompt, "analyst")
    
    def generate_strategy(self, cluster_info: Dict[str, Any]) -> str:
        """Генерация стратегии продвижения по кластерам"""
        prompt = f"""
Разработай стратегию ASO продвижения по кластерам.

📊 КЛАСТЕРЫ:
{json.dumps(cluster_info, indent=2, ensure_ascii=False)[:5000]}

На основе данных предложи:
1. Стратегию для каждого кластера с учётом правил R1-R12
2. Приоритетные кластеры для продвижения
3. Рекомендации по распределению мотива
4. Ожидаемые результаты

Формат ответа:
## Стратегия по кластерам
### [Кластер 1]
- Текущее состояние: ...
- Рекомендуемый сценарий: ...
- Действия: ...

## Приоритеты
...

## Ожидаемые результаты
...
"""
        return self._generate(prompt, "recommendations")
    
    def explain_clusters(self, cluster_stats: pd.DataFrame) -> str:
        """Объяснение кластеров с визуализацией методологии"""
        cluster_text = cluster_stats.to_string()
        prompt = f"""
Объясни результаты кластеризации ключевых слов.

📊 СТАТИСТИКА ПО КЛАСТЕРАМ:
{cluster_text}

Вопросы:
1. Что означают эти кластеры с бизнес-точки зрения?
2. Как они соотносятся с правилами методологии (R1-R12)?
3. Какие кластеры наиболее ценные?
4. Какие кластеры требуют немедленного внимания?
5. Как использовать эту информацию для ASO-стратегии?

Дай практические рекомендации по работе с каждым кластером.
"""
        return self._generate(prompt, "analyst")
    
    def explain_plot(self, plot_name: str, plot_description: str, data_context: str = "") -> str:
        """Объяснение графика с методологией"""
        prompt = f"""
Объясни график ASO-аналитики.

📈 ИНФОРМАЦИЯ О ГРАФИКЕ:
- Название: {plot_name}
- Описание: {plot_description}
- Контекст данных: {data_context if data_context else "Не указан"}

Вопросы:
1. Что показывает этот график?
2. Какие тренды или паттерны видны?
3. Как это связано с методологией ASO (кластеры, сценарии, правила)?
4. Какие выводы можно сделать для стратегии продвижения?
5. Какие следующие шаги рекомендуете?

Используй профессиональную ASO терминологию.
"""
        return self._generate(prompt, "plots")
    
    def chat(self, question: str, context: Optional[str] = None) -> str:
        """Свободный диалог с методологией"""
        context_prompt = f"""
Контекст: {context if context else "Общий ASO анализ"}

Вопрос: {question}

Используй ASO методологию для ответа. Если вопрос о графиках — дай детальную интерпретацию.
Если вопрос о стратегии — дай конкретные рекомендации с учётом правил R1-R12.
"""
        return self._generate(context_prompt, "analyst")
    
    def get_model_explanation(self, model_results: Dict[str, Any]) -> str:
        """Объяснение результатов ML модели"""
        prompt = f"""
Объясни результаты ML модели для прогнозирования изменения позиции.

📊 РЕЗУЛЬТАТЫ МОДЕЛИ:
{json.dumps(model_results, indent=2, ensure_ascii=False)}

Вопросы:
1. Что означают метрики MAE, RMSE, R²?
2. Какие признаки наиболее важны и почему?
3. Как это согласуется с ASO методологией?
4. Какие практические выводы можно сделать?
5. Как улучшить модель?

Дай интерпретацию для бизнес-пользователя.
"""
        return self._generate(prompt, "analyst")
    
    def get_recommendations(self, data_summary: Dict[str, Any]) -> str:
        """Получение рекомендаций на основе всех данных"""
        prompt = f"""
Дай конкретные ASO-рекомендации на основе данных.

📊 ДАННЫЕ:
{json.dumps(data_summary, indent=2, ensure_ascii=False)}

Сформулируй рекомендации по категориям:
1. 🔴 КРИТИЧЕСКИЕ (срочные действия)
2. 🟡 ВАЖНЫЕ (ближайшие действия)
3. 🟢 РЕКОМЕНДУЕМЫЕ (долгосрочные действия)

Для каждой рекомендации укажи:
- Что делать
- Почему это важно (связь с методологией)
- Ожидаемый эффект
- Сроки

Используй правила R1-R12 для обоснования.
"""
        return self._generate(prompt, "recommendations")
    
    def get_status(self) -> dict:
        """Получение статуса LLM"""
        return {
            'is_available': self.is_available,
            'model': self.model,
            'url': self.base_url,
            'methodology_version': '1.0'
        }