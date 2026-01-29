# 多 Agent 开发框架对比分析

> 版本: 1.0 | 日期: 2026-01-28

## 一、业界最佳实践总结

### 1.1 核心设计模式

基于搜索结果，2026 年多 Agent 系统的主流设计模式包括：

| 模式 | 描述 | 来源 |
|------|------|------|
| **Orchestrator-Worker** | 主 Agent 协调 + 多个专业化子 Agent 并行执行 | Anthropic, AutoGen |
| **角色专业化** | 每个 Agent 专注特定领域（PM、Coder、Tester 等） | CrewAI, MetaGPT |
| **上下文隔离** | 子 Agent 拥有独立上下文窗口，避免溢出 | Claude Code |
| **文件系统状态** | 使用文件系统持久化状态，跨 Agent 共享 | LangGraph |
| **渐进式工具暴露** | 按需加载工具定义，节省上下文 | Agent Design Patterns |
| **Prompt 缓存** | 缓存命中率是生产 Agent 最重要指标 | Anthropic |

### 1.2 Anthropic 官方多 Agent 系统经验

来自 [Anthropic 工程博客](https://www.anthropic.com/engineering/multi-agent-research-system)：

**关键发现：**
- 多 Agent (Opus 4 lead + Sonnet 4 subagents) 比单 Agent **提升 90.2%** 性能
- Token 使用量解释了 **80%** 的性能差异
- 并行执行可减少 **90%** 的复杂查询时间

**最佳实践：**
1. 详细的任务委派 - 模糊指令导致子 Agent 重复工作
2. 努力程度规则 - 防止简单任务过度投入
3. 工具描述清晰 - 模糊描述导致 Agent 走错路径
4. 人工评估必要 - 自动测试遗漏边缘情况

**挑战：**
- 早期 Agent 对简单查询生成 50 个子 Agent
- 长时间运行需要持久化执行和错误恢复
- 非确定性行为导致调试困难

### 1.3 主流框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **AutoGen** (Microsoft) | 对话式协作，灵活路由 | 研究、原型 |
| **LangChain/LangGraph** | 模块化组件，状态管理高效 | 复杂工作流 |
| **CrewAI** | 角色导向，可视化设计 | 生产系统 |
| **MetaGPT** | 模拟人类团队，代码导向 | 应用开发 |

根据基准测试，LangGraph 执行速度最快，AutoGen 和 LangChain 在 Token 效率上有 8-9 倍差异。

## 二、我们框架的对比分析

### 2.1 架构对比

| 维度 | 业界最佳实践 | 我们的框架 v2.0 | 评估 |
|------|--------------|-----------------|------|
| **编排模式** | Orchestrator-Worker | Coordinator + 三层九角色 | ✅ 对齐 |
| **角色专业化** | PM, Coder, Tester 等 | 9 个专业角色 | ✅ 对齐 |
| **模型分层** | Lead 用强模型，Worker 用快模型 | Opus(决策) + Sonnet(执行) + Haiku(守护) | ✅ 对齐 |
| **上下文隔离** | 子 Agent 独立上下文 | 每个 Agent 独立执行 | ✅ 对齐 |
| **并行执行** | 3-5 子 Agent 并发 | `run_in_background=true` | ✅ 支持 |
| **质量门禁** | Hooks, 自动检查 | 4 个 Hook 脚本 | ✅ 对齐 |
| **评估系统** | 人工 + 自动评估 | Secretary 评估 + 月度总结 | ✅ 对齐 |
| **状态持久化** | 文件系统共享状态 | docs/*.md 日志文件 | ✅ 对齐 |
| **渐进式工具** | 按需加载工具 | ⚠️ 未明确实现 | ⚠️ 待改进 |
| **Prompt 缓存** | 缓存命中率优化 | ⚠️ 未明确实现 | ⚠️ 待改进 |
| **努力程度规则** | 防止过度投入 | ⚠️ 未明确实现 | ⚠️ 待改进 |
| **错误恢复** | 持久化执行 + 重试 | ⚠️ 依赖 Claude Code 内置 | ⚠️ 待改进 |

### 2.2 优势分析

**1. 三层架构设计合理** ✅
```
决策层 (Architect, PM, Secretary)  ← 高价值判断，使用 Opus
执行层 (Coder, Tester, Reviewer)   ← 核心实现，使用 Sonnet
支撑层 (Docs, Ops, Guard)          ← 辅助保障，使用 Sonnet/Haiku
```
与 Anthropic "Lead + Subagents" 模式完全对齐。

**2. Secretary 评估机制独特** ✅
- 业界少有的 Agent 表现评估系统
- 记录决策、评估表现、月度总结
- 支持持续改进

**3. 产品经理角色稀缺** ✅
- 多数框架只有技术角色
- PM 把关产品竞争力和用户价值
- 类似 MetaGPT 的 ProductManager 角色

**4. Hook 质量门禁完善** ✅
- pre-edit: 保护文件检查
- post-edit: 自动格式化
- pre-bash: 危险命令拦截
- quality-gate: 任务完成检查

### 2.3 待改进领域

**1. 渐进式工具暴露** ⚠️

当前：所有工具一次性加载
建议：
```markdown
# 在 Agent 定义中添加
tools_on_demand:
  - name: WebSearch
    trigger: "需要搜索最新信息时"
  - name: Bash
    trigger: "需要执行系统命令时"
```

**2. 努力程度规则** ⚠️

Anthropic 发现没有明确的努力程度规则会导致过度投入。

建议在 Agent 定义中添加：
```markdown
## 努力程度规则

| 任务复杂度 | 最大 Token | 最大子 Agent |
|------------|-----------|--------------|
| 简单 | 5K | 0 |
| 中等 | 20K | 2 |
| 复杂 | 100K | 5 |
```

**3. 错误恢复机制** ⚠️

当前依赖 Claude Code 内置机制。

建议添加：
```markdown
## 错误恢复策略

1. 自动重试: 网络错误最多重试 3 次
2. 状态检查点: 每完成子任务保存状态
3. 回滚机制: 失败时回滚到上一检查点
```

**4. Prompt 缓存策略** ⚠️

建议在 settings.json 中添加：
```json
{
  "prompt_caching": {
    "enabled": true,
    "cache_system_prompts": true,
    "cache_agent_definitions": true
  }
}
```

## 三、改进建议

### 3.1 短期改进 (v2.1)

| 改进项 | 优先级 | 工作量 |
|--------|--------|--------|
| 添加努力程度规则到各 Agent | 高 | 小 |
| 添加错误恢复指南 | 高 | 小 |
| 优化工具描述清晰度 | 中 | 中 |

### 3.2 中期改进 (v3.0)

| 改进项 | 优先级 | 工作量 |
|--------|--------|--------|
| 实现渐进式工具暴露 | 中 | 大 |
| 添加 Prompt 缓存配置 | 中 | 中 |
| Agent 协作协议标准化 | 中 | 大 |

### 3.3 架构演进方向

```
v2.0 (当前)              v3.0 (目标)
├── 9 角色静态定义        ├── 动态角色组合
├── 手动触发              ├── 智能任务路由
├── 独立执行              ├── 协作协议
└── 文件日志              └── 结构化状态管理
```

## 四、结论

### 4.1 总体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 架构设计 | 9/10 | 三层九角色与业界对齐 |
| 角色覆盖 | 9/10 | PM、Secretary 角色独特 |
| 质量保障 | 8/10 | Hook 系统完善 |
| 执行效率 | 7/10 | 缺少努力程度规则 |
| 错误处理 | 6/10 | 依赖内置机制 |
| 可扩展性 | 8/10 | 模块化设计良好 |
| **综合** | **8.0/10** | **接近业界最佳实践** |

### 4.2 核心结论

1. **整体架构合理** - 与 Anthropic、CrewAI、MetaGPT 的设计思路一致
2. **角色设计独特** - Secretary 评估机制和 PM 产品把关在业界少见
3. **待补充细节** - 努力程度规则、错误恢复、渐进式工具等需要完善
4. **可投入使用** - 当前版本已具备生产可用性

### 4.3 与竞品对比定位

```
功能完整度
    ↑
    │         ┌─────────┐
    │         │ MetaGPT │
    │    ┌────┴─────────┴────┐
    │    │   Our v2.0        │  ← 接近 MetaGPT
    │    └───────────────────┘
    │         ┌─────────┐
    │         │ CrewAI  │
    │    ┌────┴─────────┴────┐
    │    │   AutoGen         │
    │    └───────────────────┘
    └────────────────────────────→ 易用性
```

---

## 参考资料

- [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Agent Design Patterns](https://rlancemartin.github.io/2026/01/09/agent_design/)
- [Claude Code Subagents Documentation](https://code.claude.com/docs/en/sub-agents)
- [Top Multi-Agent AI Frameworks](https://www.multimodal.dev/post/best-multi-agent-ai-frameworks)
- [AutoGen vs LangChain Comparison](https://kanerika.com/blogs/autogen-vs-langchain/)
- [LLM Orchestration Frameworks](https://research.aimultiple.com/llm-orchestration/)

---

*分析完成于 2026-01-28*
