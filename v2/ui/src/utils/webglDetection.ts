/**
 * WebGL支持检测工具
 */

export interface WebGLInfo {
  supported: boolean;
  version: string;
  renderer: string;
  vendor: string;
  maxTextureSize: number;
  extensions: string[];
  error?: string;
}

export function detectWebGL(): WebGLInfo {
  try {
    // 创建canvas元素
    const canvas = document.createElement('canvas');
    
    // 尝试获取WebGL2上下文
    let gl = canvas.getContext('webgl2');
    let version = 'WebGL 2.0';
    
    // 如果WebGL2不支持，尝试WebGL1
    if (!gl) {
      gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      version = 'WebGL 1.0';
    }
    
    if (!gl) {
      return {
        supported: false,
        version: 'None',
        renderer: 'Unknown',
        vendor: 'Unknown',
        maxTextureSize: 0,
        extensions: [],
        error: 'WebGL not supported by browser'
      };
    }
    
    // 获取渲染器信息
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    const renderer = debugInfo ? 
      gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 
      gl.getParameter(gl.RENDERER);
    const vendor = debugInfo ? 
      gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) : 
      gl.getParameter(gl.VENDOR);
    
    // 获取最大纹理尺寸
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    
    // 获取支持的扩展
    const extensions = gl.getSupportedExtensions() || [];
    
    return {
      supported: true,
      version,
      renderer,
      vendor,
      maxTextureSize,
      extensions,
    };
  } catch (error) {
    return {
      supported: false,
      version: 'None',
      renderer: 'Unknown',
      vendor: 'Unknown',
      maxTextureSize: 0,
      extensions: [],
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

export function logWebGLInfo(): void {
  const info = detectWebGL();
  console.log('WebGL Detection Results:', info);
  
  if (!info.supported) {
    console.error('WebGL is not supported!', info.error);
  } else {
    console.log(`WebGL ${info.version} is supported`);
    console.log(`Renderer: ${info.renderer}`);
    console.log(`Vendor: ${info.vendor}`);
    console.log(`Max Texture Size: ${info.maxTextureSize}`);
    console.log(`Extensions: ${info.extensions.length} available`);
  }
}