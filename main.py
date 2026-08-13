import os
from core.llm_client import DeepSeekClient
from core.agent import SimpleReActAgent

def main():
    # 1. 初始化客户端（记得在终端执行: export DEEPSEEK_API_KEY="sk-..."）
    try:
        llm = DeepSeekClient()
    except ValueError as e:
        print(f"❌ {e}")
        print("请先设置环境变量：")
        print("  Windows: set DEEPSEEK_API_KEY=你的密钥")
        print("  Mac/Linux: export DEEPSEEK_API_KEY=你的密钥")
        return

    # 2. 初始化 Agent
    agent = SimpleReActAgent(llm)
    
    print("🤖 智能研究助理 SRA (Phase 1) 已启动")
    print("提示：输入 'exit' 退出程序\n")
    
    while True:
        user_input = input("👤 你: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # 3. 运行 Agent
        response = agent.run(user_input)
        print(f"🤖 助理: {response}\n")

if __name__ == "__main__":
    main()