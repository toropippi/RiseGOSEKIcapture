//
//		HSP3.0 plugin sample
//		onion software/onitama 2004/9
//      edit -> toropippi 2021/4
//

#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "hsp3plugin.h"
#include <omp.h> 
using namespace std;
using vi = vector<int>; // intの1次元の型に vi という別名をつける
using vvi = vector<vi>; // intの2次元の型に vvi という別名をつける
using vvvi = vector<vvi>; // intの2次元の型に vvi という別名をつける

#define BUFTYPE int

//////100枚のやつ
//skillの全100種類の画像のドットデータ=テンプレート
const int SKILLSY = 24;
BUFTYPE buffer[100][SKILLSY][192];

//文字数でスコア補正
int STRLENALL[] = { 3,7,5,5,6,6,7,6,5,6,4,3,3,4,7,3,7,6,4,4,4,7,4,3,3,2,4,2,3,2,3,5,2,4,4,7,4,2,4,2,1,4,3,2,7,3,6,4,5,3,4,4,4,7,2,7,2,4,4,3,3,3,7,4,5,7,6,5,3,4,3,3,6,7,6,6,6,4,4,7,3,2,4,5,4,4,2,4,2,6,4,3,2,7,3,7,3,4,2,4 };
float STRLENALLf[] = { 3,7,5,5,6,6,7,6,5,6,4,3,3,4,7,3,7,6,4,4,4,7,4,3,3,2,4,2,3,2,3,5,2,4,4,7,4,2,4,2,1,4,3,2,7,3,6,4,5,3,4,4,4,7,2,7,2,4,4,3,3,3,7,4,5,7,6,5,3,4,3,3,6,7,6,6,6,4,4,7,3,2,4,5,4,4,2,4,2,6,4,3,2,7,3,7,3,4,2,4 };

//各スキル画像の横幅
int GRPHCSSX[] = { 68,188,116,116,140,140,188,140,116,140,92,68,92,92,164,68,164,140,92,92,92,192,92,68,68,44,92,44,68,44,68,116,44,92,92,188,92,44,92,44,20,92,68,44,164,68,140,92,116,68,92,92,92,164,44,164,44,92,92,68,68,68,192,92,116,164,140,116,68,92,68,68,140,164,140,140,140,92,92,164,68,44,92,116,92,92,44,92,44,140,92,68,44,164,68,164,68,92,44,92 };
//↑の累積和、配列数が101になっている
int RUISEKIWASX[] = { 0,68,256,372,488,628,768,956,1096,1212,1352,1444,1512,1604,1696,1860,1928,2092,2232,2324,2416,2508,2700,2792,2860,2928,2972,3064,3108,3176,3220,3288,3404,3448,3540,3632,3820,3912,3956,4048,4092,4112,4204,4272,4316,4480,4548,4688,4780,4896,4964,5056,5148,5240,5404,5448,5612,5656,5748,5840,5908,5976,6044,6236,6328,6444,6608,6748,6864,6932,7024,7092,7160,7300,7464,7604,7744,7884,7976,8068,8232,8300,8344,8436,8552,8644,8736,8780,8872,8916,9056,9148,9216,9260,9424,9492,9656,9724,9816,9860,9952 };

BUFTYPE blackscore[2];//スキル認識画面での全部黒か判定
BUFTYPE whitescore[2];//スキル認識画面でのrgb差の合計値
/////////////////


//////スキルキャプチャ画像のやつ
const int BDR = 2;//ボーダー
const int CAPSX = 220 + BDR * 2;
const int CAPSY = 32 + BDR * 2;
BUFTYPE capbuffer[2][CAPSY][CAPSX];
////////////////


//////スロットキャプチャ画像のやつ
const int SLTSX = 132 + BDR * 2;
const int SLTSY = 44 + BDR * 2;
BUFTYPE capbufferSLT[SLTSY][SLTSX];
////////////////

//その他サイズ自由テンプレート。護石の名前やスロット画像など
vvvi buffer2 = vvvi(8);







static void omptest(void) 
{
	//mrefのデータ。bgrbgrbgrbgrbgr・・・の順で左下から埋まっている
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	int* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (int*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。
	ptr2[0] = omp_get_max_threads();
	//printf("使用可能な最大スレッド数：%d\n", );
}



//初回ロード100枚
//int gid,array mrefarray
static void Set100TpBuf( void )
{
	int gid = code_getdi(0);		// skillのナンバー
	int xsize = GRPHCSSX[gid];
	
	//mrefのデータ。bgrbgrbgrbgrbgr・・・の順で左下から埋まっている
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。

	for (int i = 0; i < SKILLSY; i++)
	{
		for (int j = 0; j < xsize; j++)
		{
			int idx = ((SKILLSY - 1 - i) * xsize + j) * 3;
			int b = ptr2[idx    ];
			int g = ptr2[idx + 1];
			int r = ptr2[idx + 2];
			int sum = r + g + b;
			int flg = 0;
			if (sum < 150) {
				flg = 1;
			}
			if (flg >= 1) {
				sum = 0;
			}
			buffer[gid][i][j] = sum;
		}
	}
}



//毎フレームキャプチャ画像を色cutして配列に格納
//array mrefarray0,array mrefarray1
static void SetCapBuf01( void )
{
	//mrefのデータ。bgrbgrbgrbgrbgr・・・の順で左下から埋まっている
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。


	PVal* pval3;
	APTR aptr3;	//配列変数の取得
	aptr3 = code_getva(&pval3);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp3;
	unsigned char* ptr3;
	phvp3 = exinfo->HspFunc_getproc(pval3->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr3 = (unsigned char*)(phvp3->GetPtr(pval3));					//データ（pval1）の実態がある先頭ポインタを取得。

	for (int iiii = 0; iiii < 2; iiii++) 
	{
		BUFTYPE blck_sc = 0;
		BUFTYPE whte_sc = 0;
		if (iiii == 1) 
		{
			swap(ptr2, ptr3);
		}

		for (int i = 0; i < CAPSY; i++)
		{
			for (int j = 0; j < CAPSX; j++)
			{
				int idx = ((CAPSY - 1 - i) * CAPSX + j) * 3;
				BUFTYPE b = ptr2[idx];
				BUFTYPE g = ptr2[idx + 1];
				BUFTYPE r = ptr2[idx + 2];

				BUFTYPE sbmx = 0;
				BUFTYPE sum = r + g + b;

				BUFTYPE rb = abs(r - b);
				BUFTYPE bg = abs(b - g);
				BUFTYPE gr = abs(g - r);

				if (sbmx < rb)sbmx = rb;
				if (sbmx < bg)sbmx = bg;
				if (sbmx < gr)sbmx = gr;

				int flg = 0;
				if (sum < 150) {
					flg = 1;
				}

				if ((g != 149) & ((r > 60) | (b > 60) | (g > 60)))
				{
					whte_sc += sbmx * sbmx;//色rgbの差の最大
				}
				if (flg >= 1) {
					sum = 0;
				}

				capbuffer[iiii][i][j] = sum;
				blck_sc += sum * sum / 9;//黒以外のところが結構あるか
			}
		}
		whitescore[iiii] = whte_sc;
		blackscore[iiii] = blck_sc;
	}
}


//スロット画像
//毎フレームキャプチャ画像を色cutして配列に格納
//array mrefarray
static void SetCapBuf4(void)
{
	//mrefのデータ。bgrbgrbgrbgrbgr・・・の順で左下から埋まっている
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。

	for (int i = 0; i < SLTSY; i++)
	{
		for (int j = 0; j < SLTSX; j++)
		{
			int idx = ((SLTSY - 1 - i) * SLTSX + j) * 3;
			BUFTYPE b = ptr2[idx];
			BUFTYPE g = ptr2[idx + 1];
			BUFTYPE r = ptr2[idx + 2];
			BUFTYPE rb = abs(r - b);
			BUFTYPE bg = abs(b - g);
			BUFTYPE gr = abs(g - r);

			BUFTYPE sbmx = 0;
			BUFTYPE sum = r + g + b;
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
			capbufferSLT[i][j] = sum;
		}
	}
}










////////////////2つの要素をまとめてソート、前の変数が優先。flag=0で小さい順、1で大きい順
void mysort2(vi& a, vi& b) {
	if (a.size() > 0) {
		vector<pair<int, int>> p(a.size());

		for(int i=0;i< a.size();i++)
		{
			p[i].first = a[i];
			p[i].second = b[i];
		}
		sort(p.begin(), p.end());
		
		for (int i = 0; i < a.size(); i++) {
			a[i] = p[i].first;
			b[i] = p[i].second;
		}
	}
	return;
}
////////////////2同時ソートここまで

//真っ黒からどのくらい外れているか
int Match100_black(int capid, long long int cutoffv_)
{
	long long int ret = blackscore[capid];
	ret *= 91;
	if (ret > cutoffv_)ret = 2000000000;
	return (int)ret;
}

//真っ黒がない状況からどのくらい外れているか
int Match100_white(int capid, long long int cutoffv_)
{
	long long int ret = whitescore[capid];
	//これが10万以下ならかなり可能性高い、マージンを取って50万に

	ret = max((long long int)500000 - whitescore[capid], (long long int)0) * 65536;
	if (ret > cutoffv_)ret = 2000000000;
	return ret;
}

//キャプチャ画像idを入力
//キャプチャの中にテンプレート画像があるか最小二乗和の最小値と座標を出力。これを100枚全部やる
//さらに真っ黒の画面ならスキルなしということなのでそれを別途計算
//さらに全ピクセル真っ黒じゃないならこれもスキルなしということなのでそれを別途計算
//最小二乗和のカットオフ値はHSPから入力
//合計100+2のスコアをソートして出力
//int cutoffv,array outarray[204*2],              array outarray_omp[100*2]
static void Match100( void )
{
	int cutoffv_ = code_getdi(0);

	//出力格納データ。outsm,outx,outy・・・の順で埋める
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	int* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (int*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。

#pragma omp parallel for num_threads(2)
//#pragma omp parallel num_threads(2)
	for (int iiii = 0; iiii < 2; iiii++) 
	{
		//int iiii= omp_get_thread_num();
		vi score = vi(102);
//#pragma omp parallel for
		for (int gid = 0; gid < 100; gid++)
		{
			BUFTYPE outsm = 2000000000;//最終的な最小二乗和
			BUFTYPE cutoffv = (long long int)cutoffv_ * (long long int)STRLENALL[gid] / 7; 
			for (int i = 0; i < CAPSY - SKILLSY; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					//i,jの座標からみて合致するか
					BUFTYPE smsqtmp = 0;//最小二乗和

					for (int it = 0; it < SKILLSY; it++)
					{
						BUFTYPE tmp;
						for (int jt = 0; jt < GRPHCSSX[gid]; jt++)
						{
							tmp = buffer[gid][it][jt] - capbuffer[iiii][it + i][jt + j];
							smsqtmp += tmp * tmp;
						}
						if (smsqtmp > cutoffv)break;//とっとと次の座標に
					}
					if (smsqtmp > cutoffv) continue;//とっとと次の座標に
					outsm = min(outsm, smsqtmp);
				}
			}

			if (outsm >= 2000000000)
			{
				score[gid] = 2000000000;
			}
			else
			{
				score[gid] = (long long int)outsm * (long long int)7 / (long long int)STRLENALL[gid];
			}
			//ptr3[gid + iiii * 100] = omp_get_thread_num();
		}


		//100この計算が終わった
		//black
		score[100] = Match100_black(iiii, cutoffv_);
		//white
		score[101] = Match100_white(iiii, cutoffv_);

		vi idx(102);
		for (int i = 0; i < 102; i++)idx[i] = i;
		mysort2(score, idx);
		for (int i = 0; i < 102; i++) ptr2[iiii * 204 + i] = score[i];
		for (int i = 0; i < 102; i++) ptr2[iiii * 204 + i + 102] = idx[i];
	}
}




//テンプレート画像を登録 色cutして配列に格納
//int gid,int xsize,int ysize,array mrefarray
static void SetUserTpBuf(void)
{
	int gid = code_getdi(0);
	int xx = code_getdi(0);
	int yy = code_getdi(0);

	//mrefのデータ。bgrbgrbgrbgrbgr・・・の順で左下から埋まっている
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。

	buffer2[gid].resize(yy);
	for (int i = 0; i < yy; i++)
	{
		buffer2[gid][i].resize(xx);
		for (int j = 0; j < xx; j++)
		{
			int idx = ((yy - 1 - i) * xx + j) * 3;
			int b = ptr2[idx];
			int g = ptr2[idx + 1];
			int r = ptr2[idx + 2];
			int sum = r + g + b;
			buffer2[gid][i][j] = sum;
		}
	}
}




//キャプチャ画像idとテンプレート画像id(buffer2のほう)を入力
//キャプチャの中にテンプレート画像があるか、最小二乗和の最小値と座標を出力
//最小二乗和のカットオフ値はHSPから入力
//int gid,int cutoffv,array outarray[3],int leftx,int rightx
static void MatchManually(void)
{
	int gid = code_getdi(0);
	int cutoffv = code_getdi(0);
	int gxsz = buffer2[gid][0].size();
	int gysz = buffer2[gid].size();

	//出力格納データ。outsm,outx,outy・・・の順で埋める
	PVal* pval2;
	APTR aptr2;	//配列変数の取得
	aptr2 = code_getva(&pval2);//	入力変数の型と実体のポインタを取得
	HspVarProc* phvp2;
	int* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//型を処理するHspVarProc構造体へのポインタ
	ptr2 = (int*)(phvp2->GetPtr(pval2));					//データ（pval1）の実態がある先頭ポインタを取得。

	int leftx = code_getdi(0);//検索開始左
	int rightx = code_getdi(0);//検索終了右
	leftx = max(leftx, 0);
	rightx = min(SLTSX - gxsz, rightx);


	int outsm = 2000000000;//最終的な最小二乗和
	int outx = -1;//最終的なx座標
	int outy = -1;//最終的なy座標
	for (int i = 1; i < SLTSY - gysz - 1; i++)
	{
		for (int j = leftx + 1; j < rightx - 1; j++)
		{
			//i,jの座標からみて合致するか。周囲4点も検査して一番誤差が少ないのを採用してよいとする
			int smsqtmp = 0;//最小二乗和
			for (int it = 0; it < gysz; it++)
			{
				for (int jt = 0; jt < gxsz;)
				{
					int tmp = 255 * 3;
					int col = buffer2[gid][it][jt];
					tmp = min(tmp, abs(col - capbufferSLT[it + i][jt + j]));
					tmp = min(tmp, abs(col - capbufferSLT[it + i + 1][jt + j]));
					tmp = min(tmp, abs(col - capbufferSLT[it + i][jt + j + 1]));
					tmp = min(tmp, abs(col - capbufferSLT[it + i - 1][jt + j]));
					tmp = min(tmp, abs(col - capbufferSLT[it + i][jt + j - 1]));
					smsqtmp += tmp * tmp;
					jt++;
				}
				if (smsqtmp > cutoffv)break;//とっとと次の座標に
			}
			if (smsqtmp > cutoffv)continue;//とっとと次の座標に

			if (outsm > smsqtmp)
			{
				outsm = smsqtmp;
				outx = j;
				outy = i;
			}

		}
	}
	ptr2[0] = outsm;
	ptr2[1] = outx;
	ptr2[2] = outy;
}




/*------------------------------------------------------------*/

static int cmdfunc( int cmd )
{
	//		実行処理 (命令実行時に呼ばれます)
	//
	code_next();							// 次のコードを取得(最初に必ず必要です)

	switch( cmd ) {							// サブコマンドごとの分岐

	case 0x00:								// newcmd
		p1 = code_getdi( 123 );		// 整数値を取得(デフォルト123)
		stat = p1;					// システム変数statに代入
		break;

	case 0x01:
		Set100TpBuf();
		break;

	case 0x02:
		SetCapBuf01();
		break;

	case 0x03:
		Match100();
		break;

	case 0x04:
		SetUserTpBuf();
		break;

	case 0x05:
		MatchManually();
		break;

	case 0x07:
		omptest();
		break;

	case 0x08:
		SetCapBuf4();
		break;

	default:
		puterror( HSPERR_UNSUPPORTED_FUNCTION );
	}
	return RUNMODE_RUN;
}





























/*------------------------------------------------------------*/

static int ref_ival;						// 返値のための変数

static void *reffunc( int *type_res, int cmd )
{
	//		関数・システム変数の実行処理 (値の参照時に呼ばれます)
	//
	//			'('で始まるかを調べる
	//
	if ( *type != TYPE_MARK ) puterror( HSPERR_INVALID_FUNCPARAM );
	if ( *val != '(' ) puterror( HSPERR_INVALID_FUNCPARAM );
	code_next();


	switch( cmd ) {							// サブコマンドごとの分岐

	case 0x00:								// newcmd

		p1 = code_geti();				// 整数値を取得(デフォルトなし)
		ref_ival = p1 * 2;				// 返値をivalに設定
		break;

	default:
		puterror( HSPERR_UNSUPPORTED_FUNCTION );
	}

	//			'('で終わるかを調べる
	//
	if ( *type != TYPE_MARK ) puterror( HSPERR_INVALID_FUNCPARAM );
	if ( *val != ')' ) puterror( HSPERR_INVALID_FUNCPARAM );
	code_next();

	*type_res = HSPVAR_FLAG_INT;			// 返値のタイプを整数に指定する
	return (void *)&ref_ival;
}


/*------------------------------------------------------------*/

static int termfunc( int option )
{
	//		終了処理 (アプリケーション終了時に呼ばれます)
	//
	return 0;
}

/*------------------------------------------------------------*/

static int eventfunc( int event, int prm1, int prm2, void *prm3 )
{
	//		イベント処理 (HSPイベント発生時に呼ばれます)
	//
	switch( event ) {
	case HSPEVENT_GETKEY:
		{
		int *ival;
		ival = (int *)prm3;
		*ival = 123;
		return 1;
		}
	}
	return 0;
}

/*------------------------------------------------------------*/
/*
		interface
*/
/*------------------------------------------------------------*/

int WINAPI DllMain (HINSTANCE hInstance, DWORD fdwReason, PVOID pvReserved)
{
	//		DLLエントリー (何もする必要はありません)
	//
	return TRUE;
}


EXPORT void WINAPI hsp3MHRise( HSP3TYPEINFO *info )
{
	//		プラグイン初期化 (実行・終了処理を登録します)
	//
	hsp3sdk_init( info );			// SDKの初期化(最初に行なって下さい)
	info->cmdfunc = cmdfunc;		// 実行関数(cmdfunc)の登録
	info->reffunc = reffunc;		// 参照関数(reffunc)の登録
	info->termfunc = termfunc;		// 終了関数(termfunc)の登録

	/*
	//	イベントコールバックを発生させるイベント種別を設定する
    info->option = HSPEVENT_ENABLE_GETKEY;
	info->eventfunc = eventfunc;	// イベント関数(eventfunc)の登録
	*/
}



/*----------------------------------------------------------------*/
