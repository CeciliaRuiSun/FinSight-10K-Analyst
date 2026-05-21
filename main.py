import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

# 1. 加载环境变量（读取你的 API Keys）
load_dotenv()

def main():
    # 2. 初始化 LlamaParse
    # result_type="markdown" 是关键，它能把表格转成 AI 最喜欢的格式
    parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="en", # 明确指定语言
    num_workers=4, # 开启多线程，加快解析速度
    # 这是一个非常有用的参数，告诉 AI 如何理解这份文档
    parsing_instruction="This is a Tesla 10-K regulatory filing. Focus on extracting all financial tables, risk factors, and inventory-related data accurately."
)

    print("🚀 正在解析 Tesla 10-K，请稍候（这可能需要一分钟）...")
    
    # 3. 开始解析（确保你的 PDF 文件名和这里写的一样）
    # 如果你的文件名叫 tesla.pdf，请把下面改掉
    documents = parser.load_data("./tsla-10-k.pdf")

    # 4. 预览结果
    # 我们先打印前 1000 个字符，看看表格解析得怎么样
    if documents:
        print("\n✅ 解析成功！以下是文档开头的预览：\n")
        print("-" * 50)
        print(documents[0].text[:1000]) 
        print("-" * 50)
        
        # 5. 把结果存成一个文本文件，方便你慢慢看
        with open("parsed_tesla.md", "w", encoding="utf-8") as f:
            full_text = "\n\n".join([doc.text for doc in documents])
            f.write(full_text)
            print(f"✅ 已成功合并并保存共 {len(documents)} 页内容至 parsed_tesla.md")

if __name__ == "__main__":
    main()