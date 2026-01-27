# rust-diagnostics 编码规范

> 提取自 [trusted-programming/rust-diagnostics](https://github.com/trusted-programming/rust-diagnostics) 项目

## 一、项目概述

rust-diagnostics 是一个 CLI 工具，用于将 Clippy 诊断信息作为内联注释嵌入 Rust 源代码，并分析 git 历史以跟踪警告修复情况。

## 二、Rust 版本与工具链

```toml
[package]
edition = "2021"
license = "Apache-2.0"
```

- **Rust Edition**: 2021
- **许可证**: Apache-2.0

## 三、命名约定

| 类型 | 命名风格 | 示例 |
|------|----------|------|
| 函数 | snake_case | `get_diagnostics_folder`, `diagnose_all_warnings` |
| 结构体 | PascalCase | `Warning`, `Hunk`, `Args` |
| 枚举 | PascalCase | `Language` |
| 常量 | SCREAMING_SNAKE_CASE | `EMPTY_STRING` |
| 生命周期 | 单字母小写 | `'query`, `'_` |
| 模块 | snake_case | `language` |

## 四、代码组织

### 4.1 模块结构

```rust
// 显式声明模块
mod language;

// 测试模块内联定义
#[cfg(test)]
mod tests {
    // ...
}
```

### 4.2 导入组织

按逻辑分组，推荐顺序：

```rust
// 1. 外部 crate
use cargo_metadata::Message;
use serde::{Deserialize, Serialize};

// 2. 标准库
use std::collections::BTreeMap;
use std::path::PathBuf;

// 3. 本地模块
mod language;
use language::Language;
```

## 五、注释风格

### 5.1 文档注释

```rust
/// 结构体/函数的文档注释
///
/// 详细说明放在第二段
struct Warning {
    /// 字段说明
    pub message: String,
}
```

### 5.2 普通注释

```rust
// 单行注释置于代码上方
let result = process();

let value = compute(); // 或者放在行尾
```

### 5.3 命令行参数文档

```rust
#[derive(StructOpt)]
struct Args {
    /// count the number of warnings per KLOC
    #[structopt(long)]
    density: bool,
}
```

## 六、错误处理

### 6.1 推荐使用 anyhow

```rust
use anyhow::{anyhow, bail, Result};

fn process() -> Result<T> {
    // 创建错误
    return Err(anyhow!("error message"));

    // 提前返回错误
    bail!("something went wrong");
}
```

### 6.2 错误处理模式

```rust
// 链式处理
result.ok()           // 静默忽略错误转为 Option
result.unwrap()       // 假设成功（仅在确定不会失败时使用）

// 精细化处理
match result {
    Ok(value) => process(value),
    Err(e) => handle_error(e),
}
```

## 七、函数签名

### 7.1 参数类型

```rust
// 使用引用避免所有权转移
fn process(source: &[u8], config: &Config) -> Result<Output> {
    // ...
}

// 需要修改时使用可变引用
fn update(state: &mut State) {
    // ...
}
```

### 7.2 泛型约束

```rust
// 简单约束用 trait bounds
fn parse<T: FromStr>(input: &str) -> Result<T>

// 复杂约束用 where 从句
fn complex<T, U>(a: T, b: U) -> Result<()>
where
    T: Display + Debug,
    U: Iterator<Item = String>,
{
    // ...
}
```

## 八、特征实现

### 8.1 派生特征

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Warning {
    // ...
}
```

### 8.2 手动实现

```rust
impl FromStr for Language {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // ...
    }
}

impl Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ...
    }
}
```

## 九、Clippy 规则配置

项目采用严格的 Clippy 规则，以下为推荐配置：

### 9.1 文档要求

```toml
# 要求异步函数文档记录错误和 panic 情况
```

### 9.2 并发安全

```toml
# 禁止在持有锁或 RefCell 引用时进行 await
```

### 9.3 类型转换

```toml
# 严格限制显式转换
# 警告有损转换和符号丢失
```

### 9.4 代码风格

```toml
# 禁用通配符导入 (use foo::*)
# 禁止使用 dbg! 宏（生产代码）
# 警告过度内联
```

### 9.5 安全实践

```toml
# 警告使用 unwrap() 和 expect()
# 推荐使用 ? 操作符或 match 处理
```

### 9.6 Unicode 安全

```toml
# 禁止代码和注释中的文本方向控制字符
```

## 十、测试规范

### 10.1 测试组织

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // ...
    }
}
```

### 10.2 串行测试

由于共享全局状态，测试需要串行执行：

```bash
cargo test -- --test-threads=1
```

### 10.3 测试依赖

```toml
[dev-dependencies]
serial_test = "0.10.0"  # 管理共享状态
insta = "1.26.0"        # 快照测试
```

## 十一、unsafe 代码

### 11.1 使用场景

仅在与外部 C 函数交互时使用：

```rust
// 调用 tree-sitter 等 C 库
unsafe {
    tree_sitter_rust()
}
```

### 11.2 安全注释

```rust
// SAFETY: tree_sitter_rust() 是由 tree-sitter 生成的安全函数，
// 返回有效的 Language 指针
unsafe {
    tree_sitter_rust()
}
```

## 十二、依赖管理

### 12.1 核心依赖

```toml
[dependencies]
anyhow = "1.0"           # 错误处理
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tree-sitter = "0.20"     # 代码解析
git2 = "0.15"            # Git 操作
structopt = "0.3"        # CLI 参数
```

### 12.2 构建依赖

```toml
[build-dependencies]
cc = { version = "1.0", features = ["parallel"] }
```

## 十三、构建与运行

```bash
# 构建
cargo build --release

# 测试（串行执行）
cargo test -- --test-threads=1

# Clippy 检查
cargo clippy

# 安装
cargo install --path .
```
