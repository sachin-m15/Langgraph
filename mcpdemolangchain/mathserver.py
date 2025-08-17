from mcp.server.fastmcp import FastMCP

mcp= FastMCP("math")

@mcp.tool()
def add(a:int,b:int)->int:
    """Add two numbers together"""
    return a+b

@mcp.tool()
def multiply(a:int,b:int)->int:
    """Multiply two numbers together"""
    return a*b

#the transport = "stdio" argument tells the MCP server to use the standard input/output (stdin/stdout) for communication

if __name__ == "__main__":
    mcp.run(transport="stdio")
