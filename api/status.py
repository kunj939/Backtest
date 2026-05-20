"""
Vercel Serverless Function — /api/status
"""
import os, json

HEADERS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Content-Type': 'application/json',
}

def handler(request, response=None):
    if request.method == 'OPTIONS':
        return {'statusCode': 200, 'headers': HEADERS, 'body': ''}

    key = os.environ.get('GROQ_API_KEY', '')
    return {
        'statusCode': 200,
        'headers': HEADERS,
        'body': json.dumps({
            "status":          "running",
            "server":          "QuantEdge v2.1",
            "groq_key_loaded": bool(key),
            "model":           "meta-llama/llama-4-scout-17b-16e-instruct"
        })
    }
