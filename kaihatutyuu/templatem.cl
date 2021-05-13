#define SKILLSY 24
#define GRAYY 36
#define GRAYX 224
__kernel void membdwdth(__global int* a)
{
	uint gid = get_global_id(0);
	a[gid]=12345;
}


//colはx*y*3byte
__kernel void Col_to_Gray_load(__global uchar* col,int cnt,__global int* gray)
{
	uint gid = get_global_id(0);
	int b=col[gid*3  ];
	int g=col[gid*3+1];
	int r=col[gid*3+2];
	int sum = r + g + b;
	
	int flg = 0;
	if (sum < 150) {
		flg = 1;
	}
	if (flg >= 1) {
		sum = 0;
	}
	
	gray[gid+cnt*SKILLSY*192]=sum;
}

//colはSKILLSY*192*3byte
__kernel void Col_to_Gray_skl(__global uchar* col,__global uint* gray,__global int* blackscore,__global int* whitescore,int skl_no)
{
	uint gid = get_global_id(0);
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
		whte_sc = sbmx * sbmx;
		atomic_add(&whitescore[skl_no] ,whte_sc);
	}
	if (flg >= 1) 
	{
		sum = 0;
	}
	
	
	blck_sc = sum * sum / 9;//黒以外のところが結構あるか
	atomic_add(&blackscore[skl_no] ,blck_sc);
	
	
	sum=sum<<(skl_no*16);
	sum+= (skl_no==0) ? (gray[gid]&0xffff0000):(gray[gid]&0x0000ffff);
	gray[gid]=sum;
	
}



//スロット4枚のそのままロード
__kernel void Col_to_Gray(__global uchar* col,__global int* gray)
{
	uint gid = get_global_id(0);
	int b=col[gid*3  ];
	int g=col[gid*3+1];
	int r=col[gid*3+2];
	int sum = r + g + b;
	gray[gid]=sum;
}





//colはx*y*3byte
__kernel void Col_to_Gray_slot(__global uchar* col,__global int* gray)
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
	gray[gid]=sum;
}





//累積和も求めつつ、自分の座標ぎめをする。最初の1回だけ実行されるやつ
__kernel void Mypos(__global uint* myposcol,__global uint* mypos,__global int* mem_GRPHCSSX,__global uint* buffer)
{
	uint gid = get_global_id(0);
	if (gid>=10020*24)return;
	uint sm=0;
	uint lsm=0;
	uint skilno=0;
	uint x=0;
	uint y=0;
	for(int i=0;i<101;i++)
	{
		lsm=sm;
		sm+=mem_GRPHCSSX[i];
		if ((lsm*24<=gid)&(gid<sm*24))
		{
			skilno=i;
			x=gid-lsm*24;
			y=x/mem_GRPHCSSX[i];
			x%=mem_GRPHCSSX[i];
			break;
		}
	}
	
	mypos[gid]=skilno+x*256+y*65536;
	myposcol[gid]=buffer[skilno*192*SKILLSY+y*192+x];
}



















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







//テンプレートマッチング、埋め込み
//local_sizeは32固定→あとで64へ
__kernel void Match2(__global uint* GRPHCSSX,__global uint* buffer,__global uint* gray,__global uint* Sum1Result,__global uint* mem_STRLENALL)
{
	uint gid = get_global_id(0);
	uint sklid=gid/(24*12);
	uint x=gid%(24*12);//キャプ画像でのoffset
	uint y=x/24;//キャプ画像でのoffset
	x%=24;
	
	uint lpx=GRPHCSSX[sklid];
	
	uint sc0=0;
	uint sc1=0;
	for(uint i=0;i<24;i++)
	{
		for(uint j=0;j<lpx;j++)
		{
			uint g=gray[x+j+(i+y)*GRAYX];
			uint g0=g&0x0000ffff;
			uint g1=g>>16;
			uint t0=buffer[sklid*192*24+i*192+j]-g0;
			uint t1=buffer[sklid*192*24+i*192+j]-g1;
			sc0+=t0*t0;
			sc1+=t1*t1;
		}
	}
	
	
	Sum1Result[gid]=sc0/mem_STRLENALL[sklid]*7;
	Sum1Result[gid+101*24*12]=sc1/mem_STRLENALL[sklid]*7;
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
