# Prompts for an OpenAI API key and stores it in the Windows User environment.
# The key is not printed and is not written to any repo file.

[CmdletBinding()]
param(
    [string]$Name = "OPENAI_API_KEY"
)

$ErrorActionPreference = "Stop"

Write-Host "This will set $Name for your Windows user account."
Write-Host "New terminals and newly started apps will inherit it automatically."
Write-Host ""

$secureKey = Read-Host -Prompt "Paste the API key value for $Name" -AsSecureString
if ($secureKey.Length -eq 0) {
    throw "No API key was entered."
}

$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "No API key was entered."
}

if ($apiKey -notmatch "^sk-") {
    Write-Warning "The value does not start with 'sk-'. Continuing anyway."
}

[Environment]::SetEnvironmentVariable($Name, $apiKey, "User")
[Environment]::SetEnvironmentVariable($Name, $apiKey, "Process")
Set-Item -Path "Env:$Name" -Value $apiKey

Write-Host ""
Write-Host "$Name is now set for this PowerShell process and persisted for your Windows user."
Write-Host "Restart any already-running app/server processes so they inherit the new value."
