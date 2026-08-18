#!/usr/bin/env python
"""
Запуск ASO Analytics Service
"""

import uvicorn
import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 ASO Analytics Service v3.0                             ║
║                                                              ║
║   Сервис для анализа ASO данных приложения Sound Amplifier   ║
║                                                              ║
║   API документация: http://localhost:8000/docs              ║
║   Главная страница:  http://localhost:8000                  ║
║   ML страница:        http://localhost:8000/ml              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )