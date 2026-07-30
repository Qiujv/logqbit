<#
.SYNOPSIS
  使用 uv 在 Windows 上一键测试多版本 Python 兼容性
.EXAMPLE
  PS> .\test_all_python.ps1
#>

param(
    [string[]]$Versions = @("3.11","3.12")
)

$ErrorActionPreference = "Stop"

# 1. 安装 uv（若已装则跳过）
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv 未找到。"
}

# 2. 预装所有待测版本
foreach ($v in $Versions) {
    uv python install $v
}

# 3. 遍历测试
$fail = $false
foreach ($v in $Versions) {
    $venvDir = ".venv$($v.Replace('.',''))"
    Write-Host "`n==========  Testing Python $v  ==========" -ForegroundColor Yellow

    try {
        uv venv --python $v $venvDir
        uv sync --frozen --python $v --extra gui
        uv run --python $v --extra gui pytest
        Write-Host "Python $v 通过 ✅" -ForegroundColor Green
    }
    catch {
        Write-Host "Python $v 失败 ❌" -ForegroundColor Red
        Write-Host $_.Exception.Message
        $fail = $true
    }
}

# 4. 汇总
if ($fail) {
    Write-Host "`n有版本未通过，查看上方日志。" -ForegroundColor Red
    exit 1
}
else {
    Write-Host "`n所有版本均通过 🎉" -ForegroundColor Green
    exit 0
}
