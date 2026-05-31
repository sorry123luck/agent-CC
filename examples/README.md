# Examples

These examples are synthetic and do not contain real screenshots, messages, or local window data.

## Flow: Observe -> Query -> Preflight

1. Use `/api/v1/windows` to list visible windows and pick a target `hwnd`.
2. Send [api/observe-request.json](api/observe-request.json) to `/api/v1/observe`.
3. Use the returned `canvas_id` in [api/query-request.json](api/query-request.json).
4. Use a selected `candidate_id` in [api/act-preflight-request.json](api/act-preflight-request.json).
5. Execute only if the policy is acceptable to the caller.

## PowerShell

```powershell
$observe = Get-Content examples\api\observe-request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/observe -ContentType application/json -Body $observe
```

## Python

```python
import requests

observe = {
    "hwnd": 123456,
    "include_screenshot": False,
    "allow_vlm": False,
}

response = requests.post("http://127.0.0.1:8000/api/v1/observe", json=observe, timeout=30)
response.raise_for_status()
canvas = response.json()
print(canvas["canvas_id"])
```

