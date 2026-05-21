import os
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset

# 1. 导入 Ragas 评估指标
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy

# 导入 Langchain 中的 Google Gemini 模型
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def main():
    # 从 .env 文件加载环境变量
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError("未找到 GOOGLE_API_KEY，请检查项目目录下的 .env 文件！")

    print("初始化 Ragas 的 Critic LLM (Gemini) 和 Embeddings 模型...")
    # 2. 配置 Gemini 2.5 Flash 作为 Ragas 的评估模型 (Critic LLM)
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        timeout=120
    )
    
    # Ragas 的 answer_relevancy 指标需要用到 Embeddings，因此这里也初始化一个 Embedding 模型
    # 使用 Google AI Studio 的标准 gemini-embedding-001 模型
    eval_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key,
        timeout=120
    )

    # 3. 从 eval_dataset 导入包含 5 个 question、contexts 和 answer 的真实样本数据集
    from eval_dataset import data_samples

    # 将字典转换为 Hugging Face 的 Dataset 格式，这是 Ragas evaluate 函数必需的标准格式
    dataset = Dataset.from_dict(data_samples)
    print("\n✅ 模拟测试数据集已准备就绪。")

    # 4. 运行评估
    print("\n🚀 正在运行 Ragas 评估 (指标: Faithfulness & Answer Relevancy)...")
    
    # 配置运行参数，限制并发为 1（完全串行）以彻底防范 API 超时和并发限制
    run_config = RunConfig(timeout=300, max_workers=1)

    # 传入需要计算的指标列表、大语言模型实例以及嵌入模型实例
    # raise_exceptions=True 用于在出错时抛出异常以利于排查诊断
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
        raise_exceptions=True
    )

    # 将 Ragas 评估结果对象转化为 Pandas DataFrame 以便后续操作（如保存 CSV 或展示）
    df = result.to_pandas()
    
    print("\n📊 评估结果明细 (Pandas DataFrame):")
    # 打印包含单条提问各项得分的数据框
    print(df.to_markdown(index=False))
    
    print("\n📈 整体量化得分 (Aggregate Scores):")
    print(result)

if __name__ == "__main__":
    main()
