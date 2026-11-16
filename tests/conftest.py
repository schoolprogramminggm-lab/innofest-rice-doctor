# Добавляем корень проекта в пути поиска, чтобы тесты видели папки bot/ и model/
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
