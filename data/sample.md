# Redis 数据库简介

Redis 是一个开源的内存数据结构存储系统，常用作数据库、缓存和消息代理。

## 持久化机制
Redis 支持两种持久化方式：
1. RDB（Redis Database）：在指定时间间隔内生成数据集的时间点快照。
2. AOF（Append Only File）：记录服务器执行的所有写操作命令，并在服务器启动时重新执行这些命令来恢复数据。

## 性能特点
Redis 基于内存操作，读写速度极快，官方基准测试显示读操作可达 110000 次/秒，写操作可达 81000 次/秒。

## 常见使用场景
- 会话缓存（Session Cache）
- 排行榜（Leaderboards）
- 计数器（Counters）
- 消息队列（Message Queues）