#if OPENGL
	#define SV_POSITION POSITION
	#define VS_SHADERMODEL vs_3_0
	#define PS_SHADERMODEL ps_3_0
#else
	#define VS_SHADERMODEL vs_5_0
	#define PS_SHADERMODEL ps_5_0
#endif

Texture2D SpriteTexture;

sampler2D SpriteTextureSampler = sampler_state
{
	Texture = <SpriteTexture>;
};

struct VertexShaderOutput
{
	float4 Position : SV_POSITION;
	float4 Color : COLOR0;
	float2 TextureCoordinates : TEXCOORD0;
};

float4 MainPS(VertexShaderOutput input) : COLOR
{
	float4 OUT = tex2D(SpriteTextureSampler,input.TextureCoordinates);
	if (OUT.r > 0.65f)
		OUT = float4(0.098f, 0.231f, 0.901f, 1);
	else
		OUT = float4(0, 0, 0, 0);
	return OUT;
}

technique SpriteDrawing
{
	pass P0
	{
		AlphaBlendEnable = true;
		PixelShader = compile PS_SHADERMODEL MainPS();
	}
};