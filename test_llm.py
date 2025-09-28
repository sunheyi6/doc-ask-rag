from langchain_community.llms import Tongyi
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建大模型实例
llm = Tongyi(
    model="qwen-plus",  # 也可以用 qwen-turbo（更快更便宜）
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 测试提问
response = llm.invoke("你好，请介绍一下你自己。")
print(response)