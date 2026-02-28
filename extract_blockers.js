const fs = require('fs');
const readline = require('readline');

(async () => {
  const rl = readline.createInterface({
    input: fs.createReadStream('C:/Users/LENOVO/.claude/projects/I--ai-automation-projects-deep-neural-network-learning/4e705782-4963-451e-a5c0-c91a28408011.jsonl'),
    crlfDelay: Infinity
  });

  const tasks = [];
  for await (const line of rl) {
    try {
      const data = JSON.parse(line);
      const content = data.message?.content || [];
      for (const item of content) {
        if (item.type === 'tool_use' && item.name === 'Bash') {
          const cmd = item.input?.command || '';
          if (cmd.includes('octie create') && cmd.includes('--title')) {
            const titleMatch = cmd.match(/--title "([^"]+)"/);
            const blockerMatch = cmd.match(/--blockers ([a-z0-9,-]+)/);
            const idMatch = cmd.match(/--title[^|]+\|\s*ID[^`]*`([a-z0-9-]+)`/);
            if (titleMatch) {
              tasks.push({
                title: titleMatch[1],
                blockers: blockerMatch ? blockerMatch[1] : 'none'
              });
            }
          }
        }
      }
    } catch {}
  }

  console.log('=== Tasks with Blockers ===');
  tasks.forEach(t => {
    console.log(`${t.title} | blockers: ${t.blockers}`);
  });
})();
