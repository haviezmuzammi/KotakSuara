$ErrorActionPreference = 'Stop'
Set-Location D:\CODE_PROJECT\yassalam-tts-id
if (Get-Process voicebox* -ErrorAction SilentlyContinue) { 'VOICEBOX RUNNING - abort'; exit 1 }
$bytes = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
$env:API_KEY = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
$env:HOST = '127.0.0.1'; $env:PORT = '8781'
$env:HF_HOME = 'D:\AI\huggingface'; $env:HF_HUB_OFFLINE = '0'; $env:PKUSEG_HOME = 'D:\AI\pkuseg'
$env:YTTS_ENV_FILE = 'D:\CODE_PROJECT\yassalam-tts-id\.env.none'
$p = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList '-m','app' -PassThru -WindowStyle Hidden `
     -RedirectStandardOutput data\live_stdout.log -RedirectStandardError data\live_stderr.log
try {
  $ok = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep 1
    try { $null = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8781/health -Headers @{'X-API-Key'=$env:API_KEY}; $ok = $true; break } catch {}
  }
  if (-not $ok) { 'service did not start'; Get-Content data\live_stderr.log -Tail 30; exit 1 }
  & .venv\Scripts\python.exe scripts\live_check.py http://127.0.0.1:8781 chatterbox_ml mms_id edge_id 2>&1
  'listeners on 8781:'; Get-NetTCPConnection -LocalPort 8781 -State Listen | Select-Object LocalAddress, LocalPort | Format-Table -AutoSize | Out-String
} finally {
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  Remove-Item Env:API_KEY
}
'--- service log (tail) ---'
Get-Content data\logs\service.log -Tail 12
