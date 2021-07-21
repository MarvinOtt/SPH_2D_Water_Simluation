using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Input;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using OpenCL.Net;
using OpenCL.Net.Extensions;
using Environment = System.Environment;
using FormClosingEventArgs = System.Windows.Forms.FormClosingEventArgs;
using Form = System.Windows.Forms.Form;

namespace SPH_2D_Water_Simluation
{
	public class KB_State
	{
		byte[] data = new byte[256];
		public KB_State()
		{

		}

		[DllImport("user32.dll")]
		[return: MarshalAs(UnmanagedType.Bool)]
		private static extern bool GetKeyboardState(byte[] lpKeyState);

		public void Update()
		{
			GetKeyboardState(data);
		}

		private byte GetVirtualKeyCode(Keys key)
		{
			int value = (int)key;
			return (byte)(value & 0xFF);
		}
		public bool IsKeyDown(Keys key)
		{
			var code = GetVirtualKeyCode(key);
			if ((data[code] & 0x80) != 0)
				return true;
			return false;
		}

		public bool IsKeyUp(Keys key)
		{
			return !IsKeyDown(key);
		}
	}

	public unsafe class Game1 : Game
	{
		GraphicsDeviceManager graphics;
		SpriteBatch spriteBatch;
		SpriteFont font;
		private Event OpenCL_event;
		private RenderTarget2D metatarget, maintarget;
		private KeyboardState newkeyboardstate, oldkeyboardstate;
		public KB_State kb_state, old_kb_state;
		private Effect metaeffect;
		private Texture2D particle_tex, metatex;
		public static Random r = new Random();
		private float memorytime, kerneltime, cputime;

		private float[] averagecomputingtime = new float[1000];
		private int currenttimeindex = 0;
		private decimal averagekerneltime = 0;

		// C O N S T A N T S
		private int firstrun = 0;
		private const int MAX_PARTICLES = 200 * 200;
		private int currentparticleanz = 0;
		private int particleradius = 2;

		private int gridsizeX, gridwidth;
		private int gridsizeY, gridheight;
		private int griddepth;
		private int gridoffset = 2;

		Vector2 G = new Vector2(0.0f, 1000.0f); // external (gravitational) forces
		float REST_DENS = 150.0f; // rest density
		const float GAS_CONST = 2500.0f; // const for equation of state
		const float H = 35.0f; // kernel radius
		const float onedivH = 1 / H;
		const float HSQ = H * H; // radius^2 for optimization
		const float MASS = 65.0f; // assume all particles have the same mass
		const float VISC = 250.0f; // viscosity constant
		const float DT = 0.005f; // integration timestep
		private const float onedivDT = 1 / DT;
		private const float DTSQR = DT * DT;
		private float DTSQRhalf = DTSQR * 0.5f;
		private float DThalf = DT * 0.5f;

		public static float kStiffness = 8.0f;
		public static float kStiffnessNear = 150.0f;
		public static float kLinearViscocity = 0.01f;
		public static float kQuadraticViscocity = 0.0125f;

		int[] particelgridID;
		int[] particelgrid_curentanz;
		private float[] particlepos = new float[MAX_PARTICLES * 2];
		private float[] particlepos_buffer = new float[MAX_PARTICLES * 2];
		private float[] particlepos_prev = new float[MAX_PARTICLES * 2];
		private float[] particlespeed = new float[MAX_PARTICLES * 2];
		private float[] particlerho = new float[MAX_PARTICLES];
		private float[] particlerho_near = new float[MAX_PARTICLES];
		private float[] particlep = new float[MAX_PARTICLES];
		private float[] particlep_near = new float[MAX_PARTICLES];
		private bool[] particle_used = new bool[MAX_PARTICLES];
		byte[] particle_gridcoox = new byte[MAX_PARTICLES];
		byte[] particle_gridcooy = new byte[MAX_PARTICLES];
		int[] particle_neighbouranz = new int[MAX_PARTICLES];
		int[] particle_neighbourID = new int[MAX_PARTICLES * 2000];
		private bool IsPause = false, IsDebug = true;
		private Form thisform;

		// G P U

		private Mem GPU_particlepos, GPU_particlepos_prev, GPU_particlespeed, GPU_particlerho, GPU_particlerho_near, GPU_particlep, GPU_particlep_near, GPU_particlepos_buffer;
		private Mem GPU_particlegridID, GPU_particlegrid_curentanz, GPU_particle_gridcoox, GPU_particle_gridcooy, GPU_particle_neighbouranz, GPU_particle_neighbourID, testmem;

		private Context context;
		private Kernel testkernel, neighbourkernel, displacementkernel;
		private CommandQueue cmdqueue;
		private OpenCL.Net.Program program;
		private string program_string;

		private float height;


		public static int Screenwidth = System.Windows.Forms.Screen.PrimaryScreen.Bounds.Width;
		public static int Screenheight = System.Windows.Forms.Screen.PrimaryScreen.Bounds.Height;

		#region OPEN CL

		OpenCL.Net.Program OpenCL_CompileProgram(Context context, Device device, string Sourcecode, out string errorstring)
		{
			ErrorCode errorcode;
			OpenCL.Net.Program program;
			program = Cl.CreateProgramWithSource(context, 1, new[] { Sourcecode }, new[] { (IntPtr)Sourcecode.Length }, out errorcode);
			if (errorcode != ErrorCode.Success)
				Console.WriteLine(errorcode.ToString());
			//-cl-opt-disable
			//-cl-mad-enable
			//-cl-fast-relaxed-math
			//-cl-strict-aliasing // BEST
			errorcode = Cl.BuildProgram(program, 0, null, "-cl-strict-aliasing", null, IntPtr.Zero);
			errorstring = "";
			if (Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Status, out errorcode).CastTo<BuildStatus>() != BuildStatus.Success)
			{

				if (errorcode != ErrorCode.Success) // Couldn´t get Programm Build Info
					errorstring += "ERROR: " + "Cl.GetProgramBuildInfo" + " (" + errorcode.ToString() + ")" + "\r\n";
				errorstring += "Cl.GetProgramBuildInfo != Success" + "\r\n";
				errorstring += Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out errorcode); // Printing Log
			}
			else
				errorstring = "SUCCESS";
			return program;
		}

		ErrorCode OpenCL_RunKernel(Kernel kernel, CommandQueue queue, int size)
		{
			return Cl.EnqueueNDRangeKernel(queue, kernel, 1, null, new IntPtr[] { new IntPtr(size), }, null, 0, null, out OpenCL_event);
		}

		ErrorCode OpenCL_ReadGPUMemory<T>(CommandQueue queue, Mem GPUMemory, T[] data) where T : struct
		{
			int size = Marshal.SizeOf<T>() * data.Length;
			return Cl.EnqueueReadBuffer(queue, GPUMemory, Bool.True, IntPtr.Zero, new IntPtr(size), data, 0, null, out OpenCL_event);
		}

		Mem OpenCL_CreateGPUBuffer<T>(Context context, int length, out ErrorCode error) where T : struct
		{
			int size = Marshal.SizeOf<T>() * length;
			return (Mem)Cl.CreateBuffer(context, MemFlags.ReadWrite, size, out error);
		}
		ErrorCode OpenCL_WriteGPUMemory<T>(CommandQueue queue, Mem GPUMemory, T[] data) where T : struct
		{
			int size = Marshal.SizeOf<T>() * data.Length;
			return Cl.EnqueueWriteBuffer(queue, GPUMemory, Bool.True, IntPtr.Zero, new IntPtr(size), data, 0, null, out OpenCL_event);

		}

		Context OpenCL_CreateContext(int numdevices, Device[] devices, out ErrorCode error)
		{
			return Cl.CreateContext(null, 1, devices, null, IntPtr.Zero, out error);
		}

		#endregion

		public Game1()
		{
			graphics = new GraphicsDeviceManager(this);
			Content.RootDirectory = "Content";
		}

		protected override void Initialize()
		{
			graphics.GraphicsProfile = GraphicsProfile.HiDef;
			IsMouseVisible = true;
			var form = (System.Windows.Forms.Form)System.Windows.Forms.Control.FromHandle(this.Window.Handle);
			form.Location = new System.Drawing.Point(0, 0);
			thisform = form;
			graphics.IsFullScreen = false;
			this.IsFixedTimeStep = true;
			TargetElapsedTime = new TimeSpan((long)(TimeSpan.TicksPerMillisecond * 11.94444));
			Window.IsBorderless = true;
			graphics.SynchronizeWithVerticalRetrace = false;
			graphics.PreferredBackBufferHeight = Screenheight;
			graphics.PreferredBackBufferWidth = Screenwidth;
			graphics.ApplyChanges();
			graphics.ApplyChanges();

			// Inizialising Arrays
			griddepth = 400;
			gridwidth = gridheight = (int)(H * 1.0f);
			gridsizeX = Screenwidth / gridwidth + 5;
			gridsizeY = Screenheight / gridheight + 5;
			particelgridID = new int[gridsizeX * gridsizeY * griddepth];
			particelgrid_curentanz = new int[gridsizeX * gridsizeY];
			height = gridsizeY * gridheight - 2 * gridoffset * gridheight;
			height = 200;

			Form f = Form.FromHandle(Window.Handle) as Form;
			if (f != null) { f.FormClosing += f_FormClosing; }

			base.Initialize();
		}

		private void f_FormClosing(object sender, FormClosingEventArgs e)
		{
			this.Exit();
			Environment.Exit(0);
			Thread.Sleep(100);
			base.Exit();
		}

		protected override void LoadContent()
		{
			spriteBatch = new SpriteBatch(GraphicsDevice);
			font = Content.Load<SpriteFont>("font");
			particle_tex = Content.Load<Texture2D>("white ball");
			metatex = Content.Load<Texture2D>("meta");

			metatarget = new RenderTarget2D(GraphicsDevice, Screenwidth, Screenheight, false, SurfaceFormat.Bgra32, DepthFormat.None);
			maintarget = new RenderTarget2D(GraphicsDevice, Screenwidth, Screenheight, false, SurfaceFormat.Color, DepthFormat.None);
			metaeffect = Content.Load<Effect>("metaeffect");
			kb_state = new KB_State();
			old_kb_state = new KB_State();
			oldkeyboardstate = Keyboard.GetState();


			program_string = File.ReadAllText("main.c");

			// Inizialising GPU and Kernel
			ErrorCode err;
			string errorstring;
			Platform[] platforms = Cl.GetPlatformIDs(out err); // Getting all Platforms
			Console.WriteLine("Length:" + platforms.Length);
			Device[] devices = Cl.GetDeviceIDs(platforms[0], DeviceType.Gpu, out err); // Getting all devices
			Console.WriteLine("Length_devices0:" + devices.Length);
			context = OpenCL_CreateContext(1, devices, out err);
			cmdqueue = Cl.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, out err);
			program = OpenCL_CompileProgram(context, devices[0], program_string, out errorstring);
			if (errorstring != "SUCCESS")
			{
				Console.WriteLine(errorstring);
				throw new System.InvalidOperationException("Error during Building the Program");
			}
			else
				Console.WriteLine("Building program succeeded");

			neighbourkernel = Cl.CreateKernel(program, "Getneighbours_calcviscosity", out err);
			displacementkernel = Cl.CreateKernel(program, "GetDensityPressureDisplacement", out err);

			//Inizialising GPU Memory
			GPU_particlepos = OpenCL_CreateGPUBuffer<float>(context, particlepos.Length, out err);
			GPU_particlepos_buffer = OpenCL_CreateGPUBuffer<float>(context, particlepos_buffer.Length, out err);
			GPU_particlepos_prev = OpenCL_CreateGPUBuffer<float>(context, particlepos_prev.Length, out err);
			GPU_particlespeed = OpenCL_CreateGPUBuffer<float>(context, particlespeed.Length, out err);
			GPU_particlerho = OpenCL_CreateGPUBuffer<float>(context, particlerho.Length, out err);
			GPU_particlerho_near = OpenCL_CreateGPUBuffer<float>(context, particlerho_near.Length, out err);
			GPU_particlep = OpenCL_CreateGPUBuffer<float>(context, particlep.Length, out err);
			GPU_particlep_near = OpenCL_CreateGPUBuffer<float>(context, particlep_near.Length, out err);

			GPU_particlegrid_curentanz = OpenCL_CreateGPUBuffer<int>(context, particelgrid_curentanz.Length, out err);
			GPU_particlegridID = OpenCL_CreateGPUBuffer<int>(context, particelgridID.Length, out err);
			GPU_particle_gridcoox = OpenCL_CreateGPUBuffer<int>(context, particle_gridcoox.Length, out err);
			GPU_particle_gridcooy = OpenCL_CreateGPUBuffer<int>(context, particle_gridcooy.Length, out err);
			GPU_particle_neighbouranz = OpenCL_CreateGPUBuffer<int>(context, particle_neighbouranz.Length, out err);
			GPU_particle_neighbourID = OpenCL_CreateGPUBuffer<int>(context, particle_neighbourID.Length, out err);

			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos, particlepos);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos_prev, particlepos_prev);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlespeed, particlespeed);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particle_neighbouranz, particle_neighbouranz);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particle_neighbourID, particle_neighbourID);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particle_gridcoox, particle_gridcoox);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particle_gridcooy, particle_gridcooy);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlegrid_curentanz, particelgrid_curentanz);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlegridID, particelgridID);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlerho, particlerho);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlerho_near, particlerho_near);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlep, particlep);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlep_near, particlep_near);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos_buffer, particlepos_buffer);

			Cl.SetKernelArg(neighbourkernel, 3, GPU_particle_neighbouranz);
			Cl.SetKernelArg(neighbourkernel, 4, GPU_particle_neighbourID);
			Cl.SetKernelArg(neighbourkernel, 0, GPU_particlepos);
			Cl.SetKernelArg(neighbourkernel, 1, GPU_particlepos_prev);
			Cl.SetKernelArg(neighbourkernel, 2, GPU_particlespeed);
			Cl.SetKernelArg(neighbourkernel, 5, GPU_particle_gridcoox);
			Cl.SetKernelArg(neighbourkernel, 6, GPU_particle_gridcooy);
			Cl.SetKernelArg(neighbourkernel, 7, GPU_particlegrid_curentanz);
			Cl.SetKernelArg(neighbourkernel, 8, GPU_particlegridID);
			Cl.SetKernelArg(neighbourkernel, 9, gridsizeX);
			Cl.SetKernelArg(neighbourkernel, 10, gridsizeY);
			Cl.SetKernelArg(neighbourkernel, 11, kLinearViscocity);
			Cl.SetKernelArg(neighbourkernel, 12, kQuadraticViscocity);

			Cl.SetKernelArg(displacementkernel, 0, GPU_particlepos);
			Cl.SetKernelArg(displacementkernel, 1, GPU_particlepos_prev);
			Cl.SetKernelArg(displacementkernel, 2, GPU_particlerho);
			Cl.SetKernelArg(displacementkernel, 3, GPU_particlerho_near);
			Cl.SetKernelArg(displacementkernel, 4, GPU_particlep);
			Cl.SetKernelArg(displacementkernel, 5, GPU_particlep_near);
			Cl.SetKernelArg(displacementkernel, 6, GPU_particle_neighbouranz);
			Cl.SetKernelArg(displacementkernel, 7, GPU_particle_neighbourID);
			Cl.SetKernelArg(displacementkernel, 8, REST_DENS);
			Cl.SetKernelArg(displacementkernel, 9, kStiffness);
			Cl.SetKernelArg(displacementkernel, 10, kStiffnessNear);
			Cl.SetKernelArg(displacementkernel, 11, GPU_particlespeed);
			Cl.SetKernelArg(displacementkernel, 12, height);
			Cl.SetKernelArg(displacementkernel, 13, G.X);
			Cl.SetKernelArg(displacementkernel, 14, G.Y);
			Cl.SetKernelArg(displacementkernel, 15, Screenheight);
			Cl.SetKernelArg(displacementkernel, 16, Screenwidth);

			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlerho, particlerho);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlerho_near, particlerho_near);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlep, particlep);
			OpenCL_WriteGPUMemory(cmdqueue, GPU_particlep_near, particlep_near);



			int xsize = 150;
			int ysize = 150;


			// Particles for Dam Break Simulation
			for (int i = 0; i < xsize; i++)
			{
				for (int j = 0; j < ysize; j++)
				{
					currentparticleanz += 1;
					particle_used[i + xsize * j] = true;
					particlepos[(i + xsize * j) * 2] = particlepos_prev[(i + xsize * j) * 2] = particleradius * 2.55f * i + r.Next(0, 1000) / 10000.0f + 004;
					particlepos[(i + xsize * j) * 2 + 1] = particlepos_prev[(i + xsize * j) * 2 + 1] = particleradius * 2.55f * j + 010;
					particlespeed[(i + xsize * j) * 2] = 0;
					particlespeed[(i + xsize * j) * 2 + 1] = 0;
				}
			}
		}

		protected override void UnloadContent()
		{
		}

		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		protected override void Update(GameTime gameTime)
		{
			newkeyboardstate = Keyboard.GetState();
			kb_state.Update();
			if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
				Exit();



			#region INPUT

			if ((kb_state.IsKeyDown(Keys.LeftAlt) && kb_state.IsKeyDown(Keys.LeftControl)) || true)
			{
				if (kb_state.IsKeyDown(Keys.LeftAlt) && kb_state.IsKeyDown(Keys.LeftControl) && kb_state.IsKeyDown(Keys.Back))
				{
					thisform.Close();
				}

				if (kb_state.IsKeyDown(Keys.Space) && old_kb_state.IsKeyUp(Keys.Space))
					IsPause ^= true;
				if (kb_state.IsKeyDown(Keys.N) && old_kb_state.IsKeyUp(Keys.N))
					IsDebug ^= true;
				if (kb_state.IsKeyDown(Keys.Up) && height > -600)
				{
					height -= 4;
					Cl.SetKernelArg(displacementkernel, 12, height);
				}

				if (kb_state.IsKeyDown(Keys.Down) && height < 410)
				{
					height += 4;
					Cl.SetKernelArg(displacementkernel, 12, height);
				}

				if (kb_state.IsKeyDown(Keys.T))
				{
					kStiffness *= 1.025f;
					Cl.SetKernelArg(displacementkernel, 9, kStiffness);
				}

				if (kb_state.IsKeyDown(Keys.G))
				{
					kStiffness /= 1.025f;
					Cl.SetKernelArg(displacementkernel, 9, kStiffness);
				}

				if (kb_state.IsKeyDown(Keys.Z))
				{
					kStiffnessNear *= 1.05f;
					Cl.SetKernelArg(displacementkernel, 10, kStiffnessNear);
				}

				if (kb_state.IsKeyDown(Keys.H))
				{
					kStiffnessNear /= 1.05f;
					Cl.SetKernelArg(displacementkernel, 10, kStiffnessNear);
				}

				if (kb_state.IsKeyDown(Keys.U))
				{
					REST_DENS *= 1.05f;
					Cl.SetKernelArg(displacementkernel, 8, REST_DENS);
				}

				if (kb_state.IsKeyDown(Keys.J))
				{
					REST_DENS /= 1.05f;
					Cl.SetKernelArg(displacementkernel, 8, REST_DENS);
				}

				if (kb_state.IsKeyDown(Keys.I))
				{
					kLinearViscocity *= 1.05f;
					Cl.SetKernelArg(neighbourkernel, 11, kLinearViscocity);
				}

				if (kb_state.IsKeyDown(Keys.K))
				{
					kLinearViscocity /= 1.05f;
					Cl.SetKernelArg(neighbourkernel, 11, kLinearViscocity);
				}

				if (kb_state.IsKeyDown(Keys.O))
				{
					kQuadraticViscocity *= 1.05f;
					Cl.SetKernelArg(neighbourkernel, 12, kQuadraticViscocity);
				}

				if (kb_state.IsKeyDown(Keys.L))
				{
					kQuadraticViscocity /= 1.05f;
					Cl.SetKernelArg(neighbourkernel, 12, kQuadraticViscocity);
				}

				if (kb_state.IsKeyDown(Keys.W))
				{
					G.Y -= 10;
					Cl.SetKernelArg(displacementkernel, 14, G.Y);
				}

				if (kb_state.IsKeyDown(Keys.S))
				{
					G.Y += 10;
					Cl.SetKernelArg(displacementkernel, 14, G.Y);
				}

				if (kb_state.IsKeyDown(Keys.A))
				{
					G.X -= 10;
					Cl.SetKernelArg(displacementkernel, 13, G.X);
				}

				if (kb_state.IsKeyDown(Keys.D))
				{
					G.X += 10;
					Cl.SetKernelArg(displacementkernel, 13, G.X);
				}
			}

			#endregion

			if (!IsPause)
			{

				// Sorting Particles into the grid
				Stopwatch watch3 = new Stopwatch();
				watch3.Start();
				for (int x = 0; x < gridsizeX; ++x)
				{
					for (int y = 0; y < gridsizeY; ++y)
					{
						particelgrid_curentanz[x + y * gridsizeX] = 0;
					}
				}

				int timer = 0;
				for (int i = 0; i < MAX_PARTICLES; i++)
				{
					int i2 = i * 2;
					if (particle_used[i])
					{
						if (particlepos[i2] <= -(gridoffset - 1) * gridwidth)
							particlepos[i2] = particlepos_prev[i2] = -(gridoffset - 1) * gridwidth + 0.1f;
						if (particlepos[i2 + 1] <= -(gridoffset - 1) * gridheight)
							particlepos[i2 + 1] = particlepos_prev[i2 + 1] = -(gridoffset - 1) * gridheight + 0.1f;
						if (particlepos[i2] >= gridwidth * (gridsizeX - 2 * (gridoffset - 1) - 1))
							particlepos[i2] = particlepos_prev[i2] = gridwidth * (gridsizeX - 2 * (gridoffset - 1) - 1) - 0.1f;
						if (particlepos[i2 + 1] >= gridheight * (gridsizeY - 2 * (gridoffset - 1) - 1))
							particlepos[i2 + 1] = particlepos_prev[i2 + 1] = gridheight * (gridsizeY - 2 * (gridoffset - 1) - 1) - 0.1f;

						int xindex = (int)(particlepos[i2] / gridwidth + gridoffset);
						int yindex = (int)(particlepos[i2 + 1] / gridheight + gridoffset);
						if (particelgrid_curentanz[xindex + yindex * gridsizeX] < griddepth)
						{
							particelgridID[xindex + yindex * gridsizeX + particelgrid_curentanz[xindex + yindex * gridsizeX] * gridsizeX * gridsizeY] = i;
							particle_gridcoox[i] = (byte)xindex;
							particle_gridcooy[i] = (byte)yindex;
							particelgrid_curentanz[xindex + yindex * gridsizeX]++;
							timer++;
							if (xindex == 0 || yindex == 0 || xindex == gridsizeX - 1 || yindex == gridsizeY - 1)
							{
								int x = 0;
							}
						}

					}
				}

				watch3.Stop();
				cputime = watch3.ElapsedTicks / ((float)Stopwatch.Frequency / 1000.0f);
				// Adding Gravity
				for (int i = 0; i < currentparticleanz; i++)
				{
					if (particle_used[i] == true)
					{
						//particlespeed[i * 2] += DT * G.X;
						//particlespeed[i * 2 + 1] += DT * G.Y;
					}
				}

				if (firstrun == 1)
				{

					OpenCL_ReadGPUMemory(cmdqueue, GPU_particlerho, particlerho);
					Cl.ReleaseEvent(OpenCL_event);
				}

				//Copying Data to GPU Memory
				ErrorCode err;
				Stopwatch watch = new Stopwatch();
				watch.Start();
				////OpenCL_WriteGPUMemory(cmdqueue, GPU_particlespeed, particlespeed);
				//Cl.ReleaseEvent(OpenCL_event);
				OpenCL_WriteGPUMemory(cmdqueue, GPU_particle_gridcoox, particle_gridcoox);
				Cl.ReleaseEvent(OpenCL_event);
				OpenCL_WriteGPUMemory(cmdqueue, GPU_particle_gridcooy, particle_gridcooy);
				Cl.ReleaseEvent(OpenCL_event);
				OpenCL_WriteGPUMemory(cmdqueue, GPU_particlegridID, particelgridID);
				Cl.ReleaseEvent(OpenCL_event);
				OpenCL_WriteGPUMemory(cmdqueue, GPU_particlegrid_curentanz, particelgrid_curentanz);
				Cl.ReleaseEvent(OpenCL_event);
				watch.Stop();


				firstrun = 1;

				OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos, particlepos);
				Cl.ReleaseEvent(OpenCL_event);
				OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos_prev, particlepos_prev);
				Cl.ReleaseEvent(OpenCL_event);

				//Setting ViscosityKernel Arguments


				// Running ViscosityKernel 
				Stopwatch watch2 = new Stopwatch();
				watch2.Start();
				OpenCL_RunKernel(neighbourkernel, cmdqueue, currentparticleanz);
				Cl.Finish(cmdqueue);
				Cl.ReleaseEvent(OpenCL_event);
				watch2.Stop();

				watch.Start();
				OpenCL_ReadGPUMemory(cmdqueue, GPU_particlepos, particlepos);
				Cl.ReleaseEvent(OpenCL_event);
				OpenCL_ReadGPUMemory(cmdqueue, GPU_particlespeed, particlespeed);
				Cl.ReleaseEvent(OpenCL_event);
				//OpenCL_ReadGPUMemory(cmdqueue, GPU_particle_neighbouranz, particle_neighbouranz);
				//Cl.ReleaseEvent(OpenCL_event);
				watch.Stop();
				for (int i = 0; i < currentparticleanz; i++)
				{

					int i2 = i * 2;
					//particlepos_prev[i2] = particlepos[i2];
					//particlepos_prev[i2 + 1] = particlepos[i2 + 1];
					particlepos[i2] += particlespeed[i2] * DT;
					particlepos[i2 + 1] += particlespeed[i2 + 1] * DT;

				}

				watch.Start();
				OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos, particlepos);
				Cl.ReleaseEvent(OpenCL_event);
				//OpenCL_WriteGPUMemory(cmdqueue, GPU_particlepos_prev, particlepos_prev);
				//Cl.ReleaseEvent(OpenCL_event);
				//OpenCL_WriteGPUMemory(cmdqueue, GPU_particlespeed, particlespeed);
				//Cl.ReleaseEvent(OpenCL_event);
				watch.Stop();
				//Setting DisplacementKernel Arguments

				//OpenCL_WriteGPUMemory(cmdqueue, GPU_particlegrid_curentanz, particelgrid_curentanz);
				//Cl.ReleaseEvent(OpenCL_event);

				// Running DisplacementKernel
				watch2.Start();
				OpenCL_RunKernel(displacementkernel, cmdqueue, currentparticleanz);
				Cl.Finish(cmdqueue);
				Cl.ReleaseEvent(OpenCL_event);
				watch2.Stop();
				kerneltime = watch2.ElapsedTicks / ((float)Stopwatch.Frequency / 1000.0f);

				OpenCL_ReadGPUMemory(cmdqueue, GPU_particlepos, particlepos);
				Cl.ReleaseEvent(OpenCL_event);
				OpenCL_ReadGPUMemory(cmdqueue, GPU_particlepos_prev, particlepos_prev);
				Cl.ReleaseEvent(OpenCL_event);

				// Reading Data
				//OpenCL_ReadGPUMemory(cmdqueue, GPU_particlepos, particlepos);
				//Cl.ReleaseEvent(OpenCL_event);
				//OpenCL_ReadGPUMemory(cmdqueue, GPU_particlerho, particlerho);
				//Cl.ReleaseEvent(OpenCL_event);
				memorytime = watch.ElapsedTicks / ((float)Stopwatch.Frequency / 1000.0f);
				if (currenttimeindex < 1000)
				{
					averagecomputingtime[currenttimeindex] = kerneltime;
					currenttimeindex++;
				}
				else if (currenttimeindex == 1000)
				{
					currenttimeindex++;
					for (int i = 0; i < 1000; i++)
					{
						averagekerneltime += (decimal)averagecomputingtime[i];
					}

					averagekerneltime /= 1000m;
				}
				//OpenCL_ReadGPUMemory(cmdqueue, GPU_particlespeed, particlespeed);
				//OpenCL_ReadGPUMemory(cmdqueue, GPU_particle_neighbouranz, particle_neighbouranz);

				for (int i = 0; i < currentparticleanz; i++)
				{
					int i2 = i * 2;
					//particlespeed[i2] = (particlepos[i2] - particlepos_prev[i2])  * onedivDT;
					//particlespeed[i2 + 1] = (particlepos[i2 + 1] - particlepos_prev[i2 + 1])  * onedivDT;
				}
			}

			//Cl.Flush(cmdqueue);
			//GPU_particlespeed.Release();
			//Collision();
			oldkeyboardstate = newkeyboardstate;
			KB_State buf = old_kb_state;
			old_kb_state = kb_state;
			kb_state = buf;

			base.Update(gameTime);

		}

		private void Collision()
		{
			for (int i = 0; i < MAX_PARTICLES; i++)
			{
				if (particle_used[i])
				{
					int i2 = i * 2;
					// enforce boundary conditions
					if (particlepos[i2] < 0.0f)
					{
						float dif = particlepos[i2];
						particlespeed[i2] -= dif * 5000f * DT + G.Y * DT - 10000.0f * DT;
						particlespeed[i2] *= 1 - 25f * DT;

					}
					if (particlepos[i2] > Screenwidth / 1.1f)
					{
						float dif = Screenwidth / 1.1f - particlepos[i2];
						particlespeed[i2] += dif * 5000f * DT + G.Y * DT - 10000.0f * DT;
						particlespeed[i2] *= 1 - 25f * DT;
					}
					if (particlepos[i2 + 1] < 0.0f)
					{
						float dif = particlepos[i2 + 1] + 1;
						particlespeed[i2 + 1] -= dif * 5000f * DT + G.Y * DT - 10000.0f * DT;
						particlespeed[i2 + 1] *= 1 - 25f * DT;
					}
					if (particlepos[i2 + 1] > Screenheight / 1.25f + height)
					{
						float dif = (Screenheight / 1.25f + height) - particlepos[i2 + 1];
						particlespeed[i2 + 1] += dif * 5000f * DT + G.Y * DT - 10000.0f * DT;
						particlespeed[i2 + 1] *= 1 - 25f * DT;
					}
				}
			}
		}

		/// <param name="gameTime">Provides a snapshot of timing values.</param>
		protected override void Draw(GameTime gameTime)
		{
			if (!IsPause)
			{
				GraphicsDevice.Clear(Color.Black);
				GraphicsDevice.SetRenderTarget(maintarget);
				spriteBatch.Begin();
				for (int i = 0; i < currentparticleanz; i++)
				{
					spriteBatch.Draw(particle_tex, new Vector2(particlepos[i * 2], particlepos[i * 2 + 1]), null, new Color(new Vector3(0.15f, 0.3f, particlerho[i] * 0.025f)) * 0.9f, 0, new Vector2(50), new Vector2(2.0f / 50.0f), SpriteEffects.None, 0);
				}

				spriteBatch.End();
				GraphicsDevice.SetRenderTarget(null);
			}

			//GraphicsDevice.SetRenderTarget(metatarget);
			//GraphicsDevice.Clear(Color.Transparent);

			//GraphicsDevice.SetRenderTarget(null);
			//GraphicsDevice.Clear(Color.White);

			//spriteBatch.Begin(SpriteSortMode.Deferred, null, null, null, null, metaeffect, Matrix.Identity);
			//spriteBatch.Draw(metatarget, Vector2.Zero, Color.White);
			//spriteBatch.End();
			spriteBatch.Begin();
			spriteBatch.Draw(maintarget, Vector2.Zero, Color.White);
			spriteBatch.End();
			if (IsDebug && !IsPause)
			{
				spriteBatch.Begin();
				spriteBatch.DrawString(font, "Gravity: " + G.ToString(), new Vector2(50, 50), Color.Red);

				// SPH CONSTANS
				spriteBatch.DrawString(font, "K_Stiffness<T,G>: " + kStiffness.ToString(), new Vector2(50, 80), Color.Red);
				spriteBatch.DrawString(font, "K_Stiffness_Near<Z,H>: " + kStiffnessNear.ToString(), new Vector2(50, 110), Color.Red);
				spriteBatch.DrawString(font, "REAST_DENS<U,J>: " + REST_DENS.ToString(), new Vector2(50, 140), Color.Red);
				spriteBatch.DrawString(font, "K_Linear_Visc<I,K>: " + kLinearViscocity.ToString(), new Vector2(50, 170), Color.Red);
				spriteBatch.DrawString(font, "K_Quadratic_Visc<O,L>: " + kQuadraticViscocity.ToString(), new Vector2(50, 200), Color.Red);

				spriteBatch.DrawString(font, "Memory Time: " + memorytime.ToString(), new Vector2(50, 250), Color.Red);
				spriteBatch.DrawString(font, "Kernel Time: " + kerneltime.ToString(), new Vector2(50, 280), Color.Red);
				spriteBatch.DrawString(font, "CPU Time: " + cputime.ToString(), new Vector2(50, 310), Color.Red);
				spriteBatch.DrawString(font, "Average Kernel Time: " + averagekerneltime.ToString(), new Vector2(50, 340), Color.Red);

				spriteBatch.End();
			}
			base.Draw(gameTime);
		}
	}
}


