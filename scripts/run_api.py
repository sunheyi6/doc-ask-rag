"""
启动 FastAPI 后端的脚本
"""
import subprocess
import sys
import os


def main():
    """启动 FastAPI 服务"""
    # 确保在项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 添加到 Python 路径
    sys.path.insert(0, os.path.join(project_root, "src"))
    
    # 启动命令
    cmd = [
        sys.executable, "-m", "uvicorn",
        "rag_agent.api.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    print("🚀 启动 API 服务...")
    print("📍 API 地址: http://localhost:8000")
    print("📚 API 文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")


if __name__ == "__main__":
    main()
