import os
import sqlite3
print('cwd:', os.getcwd())
print('python:', os.sys.executable)
fn = 'cyberbullying.db'
print('db path:', os.path.abspath(fn))
print('exists:', os.path.exists(fn))
if os.path.exists(fn):
    conn = sqlite3.connect(fn)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    print('users_table:', cur.fetchone())
    try:
        cur.execute('SELECT id, username, email FROM users LIMIT 5')
        print('rows:', cur.fetchall())
    except Exception as e:
        print('query error:', e)
    conn.close()
