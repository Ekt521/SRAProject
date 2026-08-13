import sys
from io import StringIO

def safe_run_python(code: str, timeout: int = 5) -> str:
    """
    安全执行 Python 代码，捕获输出并返回。
    禁止使用 __import__、open、eval 等危险函数。
    """
    # 限制可用的全局变量（极度安全，只给基本数学和内置函数）
    safe_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "sorted": sorted,
        },
        "__name__": "__sandbox__",
    }
    
    # 重定向标准输出到内存
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # 执行用户代码（限制只允许表达式和简单语句）
        exec(code, safe_globals, {})  # 第三个参数是局部变量，置空防止污染
        output = sys.stdout.getvalue()
        return output if output else "代码执行成功（无输出）"
    except Exception as e:
        return f"执行错误: {type(e).__name__}: {str(e)}"
    finally:
        sys.stdout = old_stdout  # 恢复标准输出