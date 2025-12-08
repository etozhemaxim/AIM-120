# Файл для очистки кодировки
with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Удаляем проблемные символы
clean_content = content.encode('utf-8', 'ignore').decode('utf-8')

with open('main_clean.py', 'w', encoding='utf-8') as f:
    f.write(clean_content)

print("Файл очищен и сохранен как main_clean.py")