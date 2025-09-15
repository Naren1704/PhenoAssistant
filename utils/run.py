from .registry import auto_discover_tools, register_all_tools

def get_manager_and_proxy(env_path: str = ".env.yaml"): # we can potentially load .env.yaml here
    """
    Import agents and return (manager, user_proxy).
    """
    from agents import get_manager_and_proxy
    return get_manager_and_proxy(env_path)

def run_task(task: str, env_path: str = ".env.yaml"):
    manager, user_proxy = get_manager_and_proxy(env_path) # Build agents
    # Auto-load any decorated tools users dropped in ./functions
    # work on this to add only tools with the decorator
    auto_discover_tools()    
    register_all_tools(manager, user_proxy) # Register all collected tools with the manager
    user_proxy.initiate_chat(manager, message=task) # start conversation
