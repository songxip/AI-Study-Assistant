"""
清空Neo4j数据库脚本
用于删除所有节点和关系，清理废弃数据
"""

from neo4j_loader import Neo4jHandler
from config import Config

def clear_neo4j_database():
    """
    清空Neo4j数据库中的所有节点和关系
    """
    try:
        print("正在连接到Neo4j数据库...")
        handler = Neo4jHandler()
        handler.connect()
        
        # 获取清空前的统计
        stats_before = handler.get_statistics()
        print(f"\n清空前的数据统计：")
        print(f"  节点数: {stats_before.get('node_count', 0)}")
        print(f"  关系数: {stats_before.get('relationship_count', 0)}")
        
        # 执行清空操作
        print("\n开始清空数据库...")
        with handler.driver.session() as session:
            # 删除所有关系和节点
            result = session.run("MATCH (n) DETACH DELETE n")
            summary = result.consume()
            print(f"✓ 已删除所有节点和关系")
            
            # 删除所有索引
            print("\n正在删除索引...")
            indexes = session.run("SHOW INDEXES").data()
            for idx in indexes:
                idx_name = idx.get('name')
                if idx_name:
                    try:
                        session.run(f"DROP INDEX {idx_name} IF EXISTS")
                        print(f"  ✓ 删除索引: {idx_name}")
                    except Exception as e:
                        print(f"  ⚠ 跳过索引 {idx_name}: {e}")
            
            # 删除所有约束
            print("\n正在删除约束...")
            constraints = session.run("SHOW CONSTRAINTS").data()
            for cons in constraints:
                cons_name = cons.get('name')
                if cons_name:
                    try:
                        session.run(f"DROP CONSTRAINT {cons_name} IF EXISTS")
                        print(f"  ✓ 删除约束: {cons_name}")
                    except Exception as e:
                        print(f"  ⚠ 跳过约束 {cons_name}: {e}")
        
        # 获取清空后的统计
        stats_after = handler.get_statistics()
        print(f"\n清空后的数据统计：")
        print(f"  节点数: {stats_after.get('node_count', 0)}")
        print(f"  关系数: {stats_after.get('relationship_count', 0)}")
        
        handler.close()
        print("\n✅ Neo4j数据库已成功清空！")
        return True
        
    except Exception as e:
        print(f"\n❌ 清空数据库失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Neo4j数据库清空工具")
    print("=" * 60)
    print(f"\n数据库配置：")
    print(f"  URI: {Config.neo4j_uri}")
    print(f"  用户名: {Config.neo4j_username}")
    print(f"\n⚠️  警告：此操作将删除数据库中的所有节点和关系！")
    
    confirm = input("\n确认要清空数据库吗？(输入 yes 确认): ")
    
    if confirm.lower() == 'yes':
        success = clear_neo4j_database()
        if success:
            print("\n💡 提示：现在可以重新运行知识图谱构建功能")
    else:
        print("\n操作已取消")
