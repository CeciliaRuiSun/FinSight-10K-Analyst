import json
from datasets import Dataset

# 这是基于你刚才的运行结果提取的核心要素数据样本
data_samples = {
    "question": [
        "What specific raw materials does Tesla mention as being critical to battery cell production, and what supply chain risks are associated with them?",
        "Compare the total automotive revenues between 2023 and 2025. What is the explicit dollar amount increase?",
        "compare revenue and cost for year 2025 and 2024",
        "Review the 'Property, Plant and Equipment' or 'Business' sections. What are the production capacities listed for Gigafactory Shanghai, Berlin, and Texas as of year-end 2025?",
        "What does Tesla state about its inventory accounting method in 2025? Were there any significant inventory write-downs or provisions for obsolete inventory reported during the year?"
    ],
    "contexts": [
        [
            "Source 1: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  ### Financial Tables Unfortunately, the provided text does not contain any specific financial tables. Financial tables typically include da…",
            "Source 2: Based on the provided text from the Tesla 10-K regulatory filing, here are the extracted financial tables, risk factors, and inventory-related data:  ### Financial Tables (Note: The provided text does not contain specific financia…",
            "Source 3: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  ### Financial Tables The provided text does not contain any specific financial tables. Please provide the relevant sections or tables for e…",
            "Source 4: ### Financial Tables *Note: The provided text does not contain specific financial tables. Please refer to the complete 10-K filing for detailed financial statements such as the balance sheet, income statement, and cash flow statem…"
        ],
        [
            "Source 1: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  - Cost of automotive sales revenues also includes adjustments to warranty expense and charges to write down the carrying value of inventory…",
            "Source 2: #### Gross Margin for Energy Generation and Storage - **2025 vs. 2024** - 2025: 29.8% - 2024: 26.2% - Change: Increased due to changes in revenue and costs associated with energy generation and storage.  #### Research and Developm…",
            "Source 3: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  ### Financial Tables  #### Cost of Revenues and Gross Margin  | Description                                          | Year Ended December …",
            "Source 4: 2024 Change | 2024 vs. 2023 Change | | | | | |------------------------------------------------------|-------------------------|----------------------|----------------------|---|---|---|---| |                                       …",
            "Source 5: 2024 Change | 2024 vs. 2023 Change | | | | | |------------------------------------------------------|-------------------------|----------------------|----------------------|---|---|---|---| |                                       …"
        ],
        [
            "(API Error - Context not fully retrieved) 2025年，汽车销售收入成本较2024年减少了56亿美元，降幅为9%。这主要是由于交付量减少、单位平均成本降低以及材料成本下降，但部分被固定成本吸收减少和关税增加所抵消。",
            "(API Error - Context not fully retrieved) 服务及其他收入成本在2025年较2024年增加了16.8亿美元，增幅为17%。"
        ],
        [
            "Source 1: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  ### Financial Tables The provided text does not contain specific financial tables. Please provide the relevant sections or tables if availa…",
            "Source 2: ### Financial Tables The provided text does not contain specific financial tables. If you have access to the complete 10-K filing, please provide the sections that include financial tables for extraction.  ### Risk Factors 1. **Es…",
            "Source 3: ### Financial Tables  **1. Inventory Valuation:** - Inventories are stated at the lower of cost or net realizable value. - Cost is computed using standard cost for vehicles and energy products, approximating actual cost on a first…",
            "Source 4: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  ### Financial Tables The provided text does not contain any specific financial tables. Please provide the relevant sections or tables for e…",
            "Source 5: ### Financial Tables *Note: The provided text does not contain specific financial tables. Please refer to the complete 10-K filing for detailed financial statements such as the balance sheet, income statement, and cash flow statem…"
        ],
        [
            "Source 1: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  - Cost of automotive sales revenues also includes adjustments to warranty expense and charges to write down the carrying value of inventory…",
            "Source 2: #### Gross Margin for Energy Generation and Storage - **2025 vs. 2024** - 2025: 29.8% - 2024: 26.2% - Change: Increased due to changes in revenue and costs associated with energy generation and storage.  #### Research and Developm…",
            "Source 3: file_name: parsed_tesla.md file_path: /Users/rs/1.Project/10-k AI Project/parsed_tesla.md  ### Financial Tables  1. **Stock-Based Compensation Expense**: - For the year ended December 31, 2025: $162 million related to the 2025 CEO…"
        ]
    ],
    "answer": [
        "Tesla提及锂和镍是电池生产的关键原材料 [3]。\n\n与这些原材料相关的供应链风险包括：\n* 对锂离子电池单元供应商的依赖，任何中断都可能限制生产能力 [3]。\n* 原材料（如锂、镍）价格和供应的波动，这可能影响生产成本和盈利能力 [3]。",
        "根据提供的资料，可以计算出2023年和2025年的总汽车收入：\n\n* **2025年总汽车收入** = 2025年总汽车销售成本 + 2025年总汽车毛利润\n    = $57,165百万 [4] + $12,361百万 [5]\n    = $69,526百万\n\n* **2023年总汽车收入** = 2023年总汽车销售成本 + 2023年总汽车毛利润\n    = $66,389百万 [4] + $16,030百万 [5]\n    = $82,419百万\n\n比较2023年和2025年的总汽车收入，发现2025年的收入低于2023年。\n变化量 = 2025年总汽车收入 - 2023年总汽车收入\n= $69,526百万 - $82,419百万 = -$12,893百万\n\n因此，2023年至2025年间，总汽车收入减少了$12,893百万 [4, 5]。",
        "以下是2025年与2024年收入和成本的比较：\n\n**成本方面：**\n*   2025年，汽车销售收入成本较2024年减少了56亿美元，降幅为9% [1]。这主要是由于交付量减少、单位平均成本降低以及材料成本下降，但部分被固定成本吸收减少和关税增加所抵消 [1]。\n*   服务及其他收入成本在2025年较2024年增加了16.8亿美元，增幅为17% [1",
        "所提供的资料不足以回答关于Gigafactory上海、柏林和德克萨斯截至2025年底的生产能力问题。资料中没有包含“Property, Plant and Equipment”或“Business”等相关章节，也没有提及具体的生产能力数据或2025年的预测 [1, 2, 3, 4, 5]。",
        "根据提供的资料：\n\n特斯拉在2025年将其库存账面价值超出其估计可变现净值时，会将减记费用计入汽车销售收入成本中 [1]。此外，当报废和现有库存超出预测需求时，也会计提准备金 [1]。能源生产和存储收入成本也包括在库存账面价值超出估计可变现净值时的减记费用，以及报废和过剩库存的准备金 [2]。\n\n然而，资料中没有明确说明特斯拉在2025年使用的具体库存会计方法（例如先进先出法、后进先出法等）。同时，资料也没有提供2025年具体的重大库存减记或报废库存准备金的金额，因此无法判断这些费用是否“重大” [1, 2]。"
    ],
    "ground_truth": [
        "Tesla explicitly mentions lithium and nickel as critical raw materials. The associated risks include dependency on suppliers and fluctuations in prices and availability, which could impact production costs and profitability.",
        "The total automotive revenue decreased. Specifically, the revenue dropped by $12,893 million from 2023 ($82,419 million) to 2025 ($69,526 million).",
        "In 2025 compared to 2024, automotive sales cost decreased by $5.6 billion (9%), mainly due to lower delivery volume, lower average unit cost, and decreased material costs. Services and other cost increased by $1.68 billion (17%).",
        "The provided documents do not contain sufficient information regarding the 'Property, Plant and Equipment' or 'Business' sections to determine the production capacities for Gigafactory Shanghai, Berlin, and Texas as of year-end 2025.",
        "Tesla states that charges to write down the carrying value of inventory when it exceeds the estimated net realizable value, as well as provisions for obsolete and excess inventory, are included in the cost of revenues. However, the specific inventory accounting method (like FIFO or LIFO) and the exact amounts of any significant inventory write-downs for 2025 are not explicitly detailed in the provided excerpts."
    ]
}

# 转换为 HuggingFace Dataset 格式，这是 Ragas 评估时必需的标准格式
dataset = Dataset.from_dict(data_samples)

if __name__ == "__main__":
    print("✅ 数据集创建成功 (Dataset created successfully)！预览如下：")
    print(dataset)
    print("\n--- 第一条数据样例预览 ---")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    # =================================================================
    # 如果你想运行 Ragas 评估，取消下面代码的注释并配置 API Key 即可使用
    # =================================================================
    
    # import os
    # from ragas import evaluate
    # from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    # # 设置评估所需的 LLM 和 Embeddings 模型 (使用刚安装的 langchain-google-genai)
    # evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    # evaluator_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # print("\n🚀 正在运行 Ragas 评估...")
    # result = evaluate(
    #     dataset=dataset, 
    #     llm=evaluator_llm,
    #     embeddings=evaluator_embeddings,
    #     metrics=[
    #         context_precision,
    #         context_recall,
    #         faithfulness,
    #         answer_relevancy,
    #     ],
    # )
    # print("\n📊 评估结果 (Evaluation Results):")
    # print(result)
    # 
    # # 将结果保存为 CSV
    # df = result.to_pandas()
    # df.to_csv("evaluation_results.csv", index=False)
    # print("结果已保存到 evaluation_results.csv")
