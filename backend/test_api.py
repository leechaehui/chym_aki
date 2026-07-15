import urllib.request
import json
try:
    req = urllib.request.Request('http://localhost:8000/api/v1/incidents')
    with urllib.request.urlopen(req) as response:
        incidents = json.loads(response.read())
        for inc in incidents:
            print(f"[{inc['status']}] {inc['incident_no']} root_cause: {inc.get('root_cause')}")
except Exception as e:
    print('Error:', e)
