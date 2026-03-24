"""
启动 Web 应用的脚本
"""
import subprocess
import sys
import os


def main():
    """启动 Streamlit 应用"""
    # 确保在项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 添加到 Python 路径
    sys.path.insert(0, os.path.join(project_root, "src"))
    
    # 启动 Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "src/rag_agent/web/app.py",
        "--server.port=8501",
        "--server.headless=true"
    ]
    
    print("[启动] 智能文档问答系统...")
    print("[地址] http://localhost:8501")
    print("\n按 Ctrl+C 停止服务\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")


if __name__ == "__main__":
    main()
