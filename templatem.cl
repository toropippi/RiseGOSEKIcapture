#define SKILLSY 24
#define GRAYY 36
#define GRAYX 224





__kernel void SklLoad(__global uchar* col,int offset,__global uchar* gray,uint xsz)
{
	uint gid = get_global_id(0);
	uint x=gid%xsz;
	uint y=gid/xsz;
	uint idx=x+y*192;
	int b=col[idx*3  ];
	int g=col[idx*3+1];
	int r=col[idx*3+2];
	int sum = r + g + b;
	
	int flg = 0;
	if (sum < 150) {
		flg = 1;
	}
	if (flg >= 1) {
		sum = 0;
	}
	
	gray[gid+offset]=(uchar)(sum/3);
}



//スロット4枚のそのままロード
__kernel void SlotLoad(__global uchar* col,__global uint* gray)
{
	uint gid = get_global_id(0);
	uint x=gid%36;
	uint y=gid/36;
	
	uint ret=0;
	for(int i=0;i<4;i++)
	{
		uint idx=x+y*36+i*36*28;
		uint b=col[idx*3  ];
		uint g=col[idx*3+1];
		uint r=col[idx*3+2];
		uint sum = (r + g + b)/3;
		ret*=256;
		ret+=sum;
	}
	gray[gid]=ret;
}














//colはSKILLSY*192*3byte
__kernel void SklFrame(__global uchar* col,__global uchar* gray,__global int* blackwhitescore)
{
	uint gid = get_global_id(0);
	uint skl_no=1-gid/(224*36);
	
	int blck_sc = 0;
	int whte_sc = 0;
	int b=col[gid*3  ];
	int g=col[gid*3+1];
	int r=col[gid*3+2];
	uint sum = r + g + b;
	int sbmx = 0;
	
	int rb = abs(r - b);
	int bg = abs(b - g);
	int gr = abs(g - r);
	if (sbmx < rb)sbmx = rb;
	if (sbmx < bg)sbmx = bg;
	if (sbmx < gr)sbmx = gr;
	int flg = 0;
	if (sum < 150) {
		flg = 1;
	}
	if ((g != 149) & ((r > 60) | (b > 60) | (g > 60)))
	{
		whte_sc = sbmx * sbmx / 9;
		atomic_add(&blackwhitescore[2+skl_no] ,whte_sc);
	}
	if (flg >= 1) 
	{
		sum = 0;
	}
	blck_sc = sum * sum / 81;//黒以外のところが結構あるか
	atomic_add(&blackwhitescore[skl_no] ,blck_sc);
	
	gray[gid%(224*36)*2+skl_no]=sum / 3;
}







//colはx*y*3byte
__kernel void SlotFrame(__global uchar* col,__global uchar* gray)
{
	uint gid = get_global_id(0);
	int b=col[gid*3  ];
	int g=col[gid*3+1];
	int r=col[gid*3+2];
	int sum = r + g + b;
	int sbmx = 0;
	int rb = abs(r - b);
	int bg = abs(b - g);
	int gr = abs(g - r);
	
	if (sbmx < rb)sbmx = rb;
	if (sbmx < bg)sbmx = bg;
	if (sbmx < gr)sbmx = gr;
	int flg = 0;
	if (sum < 320) {
		flg = 1;
	}
	if (flg >= 1) {
		sum = 0;
	}
	gray[gid]=(uchar)(sum/3);
}













//テンプレートマッチング、埋め込み
//local_sizeは32固定→あとで64へ
__kernel void Match2(__global uint* GRPHCSSX4,__global uint* buffer,__global uint* gray,__global uint* Sum1Result,__global uint* mem_STRLENALL,__global uint* GRPHCSSX_sm,int cutoff)
{
	uint gid = get_global_id(0);
	uint sklid=gid/(24*12);
	uint x=gid%(24*12);//キャプ画像でのoffset
	uint y=x/24;//キャプ画像でのoffset
	x%=24;
	
	uint lpx=GRPHCSSX4[sklid];
	
	uint sc0=0;
	uint sc1=0;
	uint ofst=GRPHCSSX_sm[sklid]*6;
	uint xfg=(x%2)*16;
	uint mask=65535*(x%2);
	x/=2;
	cutoff=cutoff*mem_STRLENALL[sklid]/7;
	
	uint g,bf,g0,g1,t0,t1,bf0,ga,gb;
	for(uint i=0;i<24;i++)
	{
		for(uint j=0;j<lpx;j++)//手動ループアンロール
		{
			bf=buffer[ofst+i*lpx+j];
			uint gg=gray[x+1+j*4/2+(i+y)*GRAYX/2];
			ga=(gray[x+j*4/2+(i+y)*GRAYX/2]>>xfg)+((gg&mask)<<16);
			gb=(gg>>xfg)+((gray[x+2+j*4/2+(i+y)*GRAYX/2]&mask)<<16);
			
			
			bf0=bf&0x000000ff;
			g0=ga&0x000000ff;
			g1=(ga>>8)&0x000000ff;
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
			
			bf0=(bf>>8)&0x000000ff;
			g0=(ga>>16)&0x000000ff;
			g1=(ga>>24);
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
			
			
			
			
			bf0=(bf>>16)&0x000000ff;
			g0=gb&0x000000ff;
			g1=(gb>>8)&0x000000ff;
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
			
			bf0=bf>>24;
			g0=(gb>>16)&0x000000ff;
			g1=(gb>>24);
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
		}
		
		if ((sc0>cutoff)&(sc1>cutoff))break;
	}
	
	
	
	if (sc0>cutoff)
	{
		Sum1Result[gid]=2000000000;
	}
	else
	{
		Sum1Result[gid]=sc0*7/mem_STRLENALL[sklid];//int をオーバーフローしないことを確認済み
	}
	
	if (sc1>cutoff)
	{
		Sum1Result[gid+101*24*12]=2000000000;
	}
	else
	{
		Sum1Result[gid+101*24*12]=sc1*7/mem_STRLENALL[sklid];
	}
	
}




//Match2のoffsetマックス版
__kernel void Match2max(__global uint* GRPHCSSX4,__global uint* buffer,__global uint* gray,__global uint* Sum1Result,__global uint* mem_STRLENALL,__global uint* GRPHCSSX_sm)
{
	uint gid = get_global_id(0);
	uint sklid=gid/(24*12);
	uint x=gid%(24*12);//キャプ画像でのoffset
	uint y=x/24;//キャプ画像でのoffset
	x%=24;
	
	uint lpx=GRPHCSSX4[sklid];
	
	uint sc0=0;
	uint sc1=0;
	uint ofst=GRPHCSSX_sm[sklid]*6;
	uint xfg=(x%2)*16;
	uint mask=65535*(x%2);
	x/=2;
	
	uint g,bf,g0,g1,t0,t1,bf0,ga,gb;
	for(uint i=0;i<24;i++)
	{
		for(uint j=0;j<lpx;j++)//手動ループアンロール
		{
			bf=buffer[ofst+i*lpx+j];
			uint gg=gray[x+1+j*4/2+(i+y)*GRAYX/2];
			ga=(gray[x+j*4/2+(i+y)*GRAYX/2]>>xfg)+((gg&mask)<<16);
			gb=(gg>>xfg)+((gray[x+2+j*4/2+(i+y)*GRAYX/2]&mask)<<16);
			
			
			bf0=bf&0x000000ff;
			g0=ga&0x000000ff;
			g1=(ga>>8)&0x000000ff;
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
			
			bf0=(bf>>8)&0x000000ff;
			g0=(ga>>16)&0x000000ff;
			g1=(ga>>24);
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
			
			
			
			
			bf0=(bf>>16)&0x000000ff;
			g0=gb&0x000000ff;
			g1=(gb>>8)&0x000000ff;
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
			
			bf0=bf>>24;
			g0=(gb>>16)&0x000000ff;
			g1=(gb>>24);
			t0=bf0-g0;
			t1=bf0-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
		}
	}
	
	
	Sum1Result[gid]=sc0*7/mem_STRLENALL[sklid];//int をオーバーフローしないことを確認済み
	Sum1Result[gid+101*24*12]=sc1*7/mem_STRLENALL[sklid];
}











//最小値
//今の所local_size=32で固定
__kernel void GetMin(__global int* Sum1Result,__global int* Sum2Result)
{
	int gid = get_global_id(0);
	int sklid=gid/32;//0～101*2
	int lid=gid%32;
	int reg=Sum1Result[sklid*24*12+lid];
	for(int i=1;i<9;i++)
	{
		reg=min(Sum1Result[sklid*24*12+i*32+lid],reg);
	}
	
	__local int msum[32];
	msum[lid]=reg;
	
	
	for(int i=16;i>0;i/=2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid<i)
		{
			msum[lid]=min(msum[lid+i],msum[lid]);
		}
	}
	
	if (lid==0)
	{
		Sum2Result[sklid]=msum[0];
	}
	
	
}








/*
136 ,48の大きさから
4枚を3箇所で
16*20スライド
36*28px
*/
//global_size=16*20 *3箇所
//dslotgryは36*28のuint型
//gry8は136*48のuchar型
//テンプレートマッチング、スロットのほう
__kernel void MatchSlot(__global uint* dslotgry,__global uchar* gry8,__global uint* Sum3Result)
{
	uint gid = get_global_id(0);
	uint kasyo = gid/(16*20);
	uint x = gid%(16*20);//キャプ画像でのoffset
	uint y = x/16;//キャプ画像でのoffset
	x%=16;
	
	uint sc0=0;
	uint sc1=0;
	uint sc2=0;
	uint sc3=0;
	uint bf0,t0,t1,bfa,bfb,bfc,bfd,bfe;
	for(int i=1;i<27;i++)
	{
		for(int j=1;j<35;j++)
		{
			uint bfa=dslotgry[i*36+j];
			uint bfb=dslotgry[(i-1)*36+j];
			uint bfc=dslotgry[(i+1)*36+j];
			uint bfd=dslotgry[i*36+j-1];
			uint bfe=dslotgry[i*36+j+1];
			uint g0=gry8[(i+y)*136+j+x+42*kasyo];
			
			t1=65536;
			bf0=bfa&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfb&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfc&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfd&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfe&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			sc0+=t1;
			
			
			t1=65536;
			bf0=(bfa>>8)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfb>>8)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfc>>8)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfd>>8)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfe>>8)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			sc1+=t1;
			
			
			t1=65536;
			bf0=(bfa>>16)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfb>>16)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfc>>16)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfd>>16)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=(bfe>>16)&0x000000ff;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			sc2+=t1;
			
			
			
			t1=65536;
			bf0=bfa>>24;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfb>>24;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfc>>24;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfd>>24;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			bf0=bfe>>24;
			t0=bf0-g0;
			t1=min(t1,t0*t0);
			sc3+=t1;
		}
	}
	
	
	Sum3Result[gid+16*20*3*0]=sc0;
	Sum3Result[gid+16*20*3*1]=sc1;
	Sum3Result[gid+16*20*3*2]=sc2;
	Sum3Result[gid+16*20*3*3]=sc3;
}



/*

16*20スライド
4枚*3箇所
から4*3のスコアを求めたい
*/
//32スレッドが16*20をまとめる。つまりglobal_size=32*4*3
__kernel void SlotSum(__global int* Sum3Result,__global int* Sum4Result)
{
	int gid = get_global_id(0);
	int mk=gid/32;
	int maime=mk/3;
	int kasyo=mk%3;
	int lid=gid%32;
	
	
	int reg=Sum3Result[mk*16*20+lid];
	for(int i=lid+32;i<16*20;i+=32)
	{
		reg=min(Sum3Result[mk*16*20+i],reg);
	}
	
	__local int msum[32];
	msum[lid]=reg;
	
	for(int i=16;i>0;i/=2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid<i)
		{
			msum[lid]=min(msum[lid+i],msum[lid]);
		}
	}
	
	if (lid==0)
	{
		Sum4Result[mk]=msum[0];
	}
	
}









	/*
	for(uint i=0;i<24;i++)
	{
		for(uint j=0;j<lpx;j++)
		{
			uint g=gray[x+j+(i+y)*GRAYX];
			uint bf=buffer[sklid*192*24+i*192+j];
			//uint g0=g&0x0000ffff;
			//uint g1=g>>16;
			uint g0=g%256;
			uint g1=g>>8;
			uint t0=bf-g0;
			uint t1=bf-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
		}
	}
	*/

/*

//テンプレートマッチング、埋め込み
//local_sizeは32固定→あとで64へ
__kernel void Match(__global int* myposcol,__global uint* mypos,__global int* gray,__global uint* Sum1Result)
{
	int gid = get_global_id(0);
	if (gid>=10020*24)return;
	__local uint lsum[32];
	
	int lid=get_local_id(0);
	lsum[lid]=0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//自分posの色
	uint mycol=myposcol[gid];
	//自分の座標
	uint myp=mypos[gid];
	uint sklno=myp%256;
	uint x=(myp>>8)%256;
	uint y=(myp>>16);
	
	
	//ここからx方向28*y方向12を逐次マッチング
	uint reg[16*2];//合計値保管用のためレジスタ確保
	int idx=0;
	for(int i=0;i<12;i++)
	{
		for(int jj=0;jj<28;)
		{
			uint cc=gray[(x+jj)+(y+i)*GRAYX];
			uint c0;//スキル1
			uint c1;//スキル2を同時計算
			
			c0=(cc&0x0000ffff);//スキル1
			c1=(cc>>16);//スキル2を同時計算
			c0=(c0-mycol)*(c0-mycol);//差の二乗
			c1=(c1-mycol)*(c1-mycol);//差の二乗
			reg[(idx%16)*2  ]=c0;
			reg[(idx%16)*2+1]=c1;
			idx++;jj++;
			
			cc=gray[(x+jj)+(y+i)*GRAYX];
			c0=(cc&0x0000ffff);//スキル1
			c1=(cc>>16);//スキル2を同時計算
			c0=(c0-mycol)*(c0-mycol);//差の二乗
			c1=(c1-mycol)*(c1-mycol);//差の二乗
			reg[(idx%16)*2  ]=c0;
			reg[(idx%16)*2+1]=c1;
			idx++;jj++;
			
			cc=gray[(x+jj)+(y+i)*GRAYX];
			c0=(cc&0x0000ffff);//スキル1
			c1=(cc>>16);//スキル2を同時計算
			c0=(c0-mycol)*(c0-mycol);//差の二乗
			c1=(c1-mycol)*(c1-mycol);//差の二乗
			reg[(idx%16)*2  ]=c0;
			reg[(idx%16)*2+1]=c1;
			idx++;jj++;
			
			cc=gray[(x+jj)+(y+i)*GRAYX];
			c0=(cc&0x0000ffff);//スキル1
			c1=(cc>>16);//スキル2を同時計算
			c0=(c0-mycol)*(c0-mycol);//差の二乗
			c1=(c1-mycol)*(c1-mycol);//差の二乗
			reg[(idx%16)*2  ]=c0;
			reg[(idx%16)*2+1]=c1;
			idx++;jj++;
			
			//レジスタが全部埋まったら
			if (idx%16==0)//後でアンロールする？
			{
				barrier(CLK_LOCAL_MEM_FENCE);
				for(int k=0;k<32;k++)
				{
					//atomic_add(&lsum[(k+lid)%32] ,reg[(k+lid)%32]);//バンクコンフリクト回避
					atomic_add(&lsum[k] ,reg[k]);
					//lsum[(k+lid)%32]+=reg[(k+lid)%32];
					//barrier(CLK_LOCAL_MEM_FENCE);
				}
				barrier(CLK_LOCAL_MEM_FENCE);
				atomic_add(&Sum1Result[sklno*28*12*2+(idx-16)*2+lid] , lsum[lid]);
				//Sum1Result[sklno*28*12*2+idx/16*16*2+lid]+=lsum[lid];
				
				lsum[lid]=0;
				barrier(CLK_LOCAL_MEM_FENCE);
			}
			
		}
	}
	
}

*/
