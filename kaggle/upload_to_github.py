#!/usr/bin/env python3
"""
将SAFB项目上传至GitHub仓库的脚本
仓库地址: https://github.com/zyh-hnu/ANB.git
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# .gitignore 内容
GITIGNORE_CONTENT = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb_checkpoints

# 模型文件
*.pth
*.pt
*.pkl
*.h5
*.weights

# 数据文件
data/
*.pkl
*.pickle

# 实验结果 (可选择保留部分)
results/centralized_runs/
results/improvement_runs/
results/figures/experiment/
results/figures/report/
!results/figures/readme/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# 日志
*.log
logs/

# 环境变量
.env
.env.local

# 临时文件
*.tmp
*.temp
.cache/
"""


def run_command(cmd, cwd=None, check=True):
    """运行shell命令并返回输出"""
    print(f"执行: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd, 
        capture_output=True, 
        text=True,
        encoding='utf-8'
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {cmd}")
    return result


def ensure_gitignore(project_root: Path) -> bool:
    """确保 .gitignore 文件存在，不存在则创建"""
    gitignore_path = project_root / ".gitignore"
    
    if gitignore_path.exists():
        print(f".gitignore 已存在: {gitignore_path}")
        return False
    
    gitignore_path.write_text(GITIGNORE_CONTENT, encoding='utf-8')
    print(f"已创建 .gitignore 文件: {gitignore_path}")
    return True


def main():
    # 项目根目录
    project_root = Path(__file__).parent.parent.resolve()
    print(f"项目目录: {project_root}")
    
    # 0. 确保 .gitignore 存在
    print("\n" + "="*50)
    print("步骤0: 检查/创建 .gitignore")
    print("="*50)
    gitignore_created = ensure_gitignore(project_root)
    
    # 1. 检查git状态
    print("\n" + "="*50)
    print("步骤1: 检查git状态")
    print("="*50)
    run_command("git status", cwd=project_root)
    
    # 2. 添加所有更改
    print("\n" + "="*50)
    print("步骤2: 添加所有更改到暂存区")
    print("="*50)
    run_command("git add .", cwd=project_root)
    
    # 3. 查看将要提交的内容
    print("\n" + "="*50)
    print("步骤3: 查看将要提交的内容")
    print("="*50)
    run_command("git status", cwd=project_root)
    
    # 4. 获取用户输入的提交信息
    print("\n" + "="*50)
    default_msg = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    commit_msg = input(f"请输入提交信息 (直接回车使用默认信息: '{default_msg}'): ").strip()
    if not commit_msg:
        commit_msg = default_msg
    
    # 5. 提交更改
    print("\n" + "="*50)
    print("步骤4: 提交更改")
    print("="*50)
    run_command(f'git commit -m "{commit_msg}"', cwd=project_root, check=False)
    
    # 6. 推送到远程仓库
    print("\n" + "="*50)
    print("步骤5: 推送到远程仓库")
    print("="*50)
    
    # 获取当前分支
    result = run_command("git branch --show-current", cwd=project_root)
    current_branch = result.stdout.strip()
    
    if not current_branch:
        # 如果没有分支信息，尝试获取HEAD
        result = run_command("git rev-parse --abbrev-ref HEAD", cwd=project_root)
        current_branch = result.stdout.strip()
    
    print(f"当前分支: {current_branch}")
    
    # 推送到origin
    run_command(f"git push origin {current_branch}", cwd=project_root)
    
    print("\n" + "="*50)
    print("✅ 上传完成!")
    print("="*50)
    print(f"仓库地址: https://github.com/zyh-hnu/ANB")


if __name__ == "__main__":
    main()
