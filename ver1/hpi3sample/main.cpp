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
using vi = vector<int>; // int��1�����̌^�� vi �Ƃ����ʖ�������
using vvi = vector<vi>; // int��2�����̌^�� vvi �Ƃ����ʖ�������
using vvvi = vector<vvi>; // int��2�����̌^�� vvi �Ƃ����ʖ�������

#define BUFTYPE int

//////100���̂���B�l�|���������̂Ŏ���101
const int SKLNUM = 101;//�Ȃ����܂߂��ɂ������邩
const int SKLNUM2 = SKLNUM + 2;//�Ȃ����܂߂�
//skill�̑S100��ނ̉摜�̃h�b�g�f�[�^=�e���v���[�g
const int SKILLSY = 24;
BUFTYPE buffer[SKLNUM][SKILLSY][192];

//�������ŃX�R�A�␳
int STRLENALL[] = { 3,7,5,5,6,6,7,6,5,6,4,3,3,4,7,3,7,6,4,4,4,7,4,3,3,2,4,2,3,2,3,5,2,4,4,7,4,2,4,2,1,4,3,2,7,3,6,4,5,3,4,4,4,7,2,7,2,4,4,3,3,3,7,4,5,7,6,5,3,4,3,3,6,7,6,6,6,4,4,7,3,2,4,5,4,4,2,4,2,6,4,3,2,7,3,7,3,4,2,4,3 };
float STRLENALLf[] = { 3,7,5,5,6,6,7,6,5,6,4,3,3,4,7,3,7,6,4,4,4,7,4,3,3,2,4,2,3,2,3,5,2,4,4,7,4,2,4,2,1,4,3,2,7,3,6,4,5,3,4,4,4,7,2,7,2,4,4,3,3,3,7,4,5,7,6,5,3,4,3,3,6,7,6,6,6,4,4,7,3,2,4,5,4,4,2,4,2,6,4,3,2,7,3,7,3,4,2,4,3 };

//�e�X�L���摜�̉���
int GRPHCSSX[] = { 68,188,116,116,140,140,188,140,116,140,92,68,92,92,164,68,164,140,92,92,92,192,92,68,68,44,92,44,68,44,68,116,44,92,92,188,92,44,92,44,20,92,68,44,164,68,140,92,116,68,92,92,92,164,44,164,44,92,92,68,68,68,192,92,116,164,140,116,68,92,68,68,140,164,140,140,140,92,92,164,68,44,92,116,92,92,44,92,44,140,92,68,44,164,68,164,68,92,44,92,68 };
//���̗ݐϘa�A�z�񐔂�101�ɂȂ��Ă���
//int RUISEKIWASX[] = { 0,68,256,372,488,628,768,956,1096,1212,1352,1444,1512,1604,1696,1860,1928,2092,2232,2324,2416,2508,2700,2792,2860,2928,2972,3064,3108,3176,3220,3288,3404,3448,3540,3632,3820,3912,3956,4048,4092,4112,4204,4272,4316,4480,4548,4688,4780,4896,4964,5056,5148,5240,5404,5448,5612,5656,5748,5840,5908,5976,6044,6236,6328,6444,6608,6748,6864,6932,7024,7092,7160,7300,7464,7604,7744,7884,7976,8068,8232,8300,8344,8436,8552,8644,8736,8780,8872,8916,9056,9148,9216,9260,9424,9492,9656,9724,9816,9860,9952 };

BUFTYPE blackscore[2];//�X�L���F����ʂł̑S����������
BUFTYPE whitescore[2];//�X�L���F����ʂł�rgb���̍��v�l
/////////////////


//////�X�L���L���v�`���摜�̂��
const int BDR = 2;//�{�[�_�[
const int CAPSX = 220 + BDR * 2;
const int CAPSY = 32 + BDR * 2;
BUFTYPE capbuffer[2][CAPSY][CAPSX];
////////////////


//////�X���b�g�L���v�`���摜�̂��
const int SLTSX = 132 + BDR * 2;
const int SLTSY = 44 + BDR * 2;
BUFTYPE capbufferSLT[SLTSY][SLTSX];
////////////////

//���̑��T�C�Y���R�e���v���[�g�B��΂̖��O��X���b�g�摜�Ȃ�
vvvi buffer2 = vvvi(8);







static void omptest(void) 
{
	//mref�̃f�[�^�Bbgrbgrbgrbgrbgr�E�E�E�̏��ō������疄�܂��Ă���
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	int* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (int*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B
	ptr2[0] = omp_get_max_threads();
	//printf("�g�p�\�ȍő�X���b�h���F%d\n", );
}



//���񃍁[�h100��
//int gid,array mrefarray
static void Set100TpBuf( void )
{
	int gid = code_getdi(0);		// skill�̃i���o�[
	int xsize = GRPHCSSX[gid];
	
	//mref�̃f�[�^�Bbgrbgrbgrbgrbgr�E�E�E�̏��ō������疄�܂��Ă���
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B

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



//���t���[���L���v�`���摜��Fcut���Ĕz��Ɋi�[
//array mrefarray0,array mrefarray1
static void SetCapBuf01( void )
{
	//mref�̃f�[�^�Bbgrbgrbgrbgrbgr�E�E�E�̏��ō������疄�܂��Ă���
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B


	PVal* pval3;
	APTR aptr3;	//�z��ϐ��̎擾
	aptr3 = code_getva(&pval3);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp3;
	unsigned char* ptr3;
	phvp3 = exinfo->HspFunc_getproc(pval3->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr3 = (unsigned char*)(phvp3->GetPtr(pval3));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B

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
					whte_sc += sbmx * sbmx;//�Frgb�̍��̍ő�
				}
				if (flg >= 1) {
					sum = 0;
				}

				capbuffer[iiii][i][j] = sum;
				blck_sc += sum * sum / 9;//���ȊO�̂Ƃ��낪���\���邩
			}
		}
		whitescore[iiii] = whte_sc;
		blackscore[iiii] = blck_sc;
	}
}


//�X���b�g�摜
//���t���[���L���v�`���摜��Fcut���Ĕz��Ɋi�[
//array mrefarray
static void SetCapBuf4(void)
{
	//mref�̃f�[�^�Bbgrbgrbgrbgrbgr�E�E�E�̏��ō������疄�܂��Ă���
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B

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










////////////////2�̗v�f���܂Ƃ߂ă\�[�g�A�O�̕ϐ����D��Bflag=0�ŏ��������A1�ő傫����
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
////////////////2�����\�[�g�����܂�

//�^��������ǂ̂��炢�O��Ă��邩
int Match100_black(int capid, long long int cutoffv_)
{
	long long int ret = blackscore[capid];
	ret *= 91;
	if (ret > cutoffv_)ret = 2000000000;
	return (int)ret;
}

//�^�������Ȃ��󋵂���ǂ̂��炢�O��Ă��邩
int Match100_white(int capid, long long int cutoffv_)
{
	long long int ret = whitescore[capid];
	//���ꂪ10���ȉ��Ȃ炩�Ȃ�\�������A�}�[�W���������50����

	ret = max((long long int)500000 - whitescore[capid], (long long int)0) * 65536;
	if (ret > cutoffv_)ret = 2000000000;
	return ret;
}

//�L���v�`���摜id�����
//�L���v�`���̒��Ƀe���v���[�g�摜�����邩�ŏ����a�̍ŏ��l�ƍ��W���o�́B�����100���S�����
//����ɐ^�����̉�ʂȂ�X�L���Ȃ��Ƃ������ƂȂ̂ł����ʓr�v�Z
//����ɑS�s�N�Z���^��������Ȃ��Ȃ炱����X�L���Ȃ��Ƃ������ƂȂ̂ł����ʓr�v�Z
//�ŏ����a�̃J�b�g�I�t�l��HSP�������
//���v100+2�̃X�R�A���\�[�g���ďo��
//int cutoffv,array outarray[204*2],              array outarray_omp[100*2]
static void Match100( void )
{
	int cutoffv_ = code_getdi(0);

	//�o�͊i�[�f�[�^�Boutsm,outx,outy�E�E�E�̏��Ŗ��߂�
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	int* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (int*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B

#pragma omp parallel for num_threads(2)
//#pragma omp parallel num_threads(2)
	for (int iiii = 0; iiii < 2; iiii++) 
	{
		//int iiii= omp_get_thread_num();
		vi score = vi(SKLNUM2);
//#pragma omp parallel for
		for (int gid = 0; gid < SKLNUM; gid++)
		{
			BUFTYPE outsm = 2000000000;//�ŏI�I�ȍŏ����a
			BUFTYPE cutoffv = (long long int)cutoffv_ * (long long int)STRLENALL[gid] / 7; 
			for (int i = 0; i < CAPSY - SKILLSY; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					//i,j�̍��W����݂č��v���邩
					BUFTYPE smsqtmp = 0;//�ŏ����a

					for (int it = 0; it < SKILLSY; it++)
					{
						BUFTYPE tmp;
						for (int jt = 0; jt < GRPHCSSX[gid]; jt++)
						{
							tmp = buffer[gid][it][jt] - capbuffer[iiii][it + i][jt + j];
							smsqtmp += tmp * tmp;
						}
						if (smsqtmp > cutoffv)break;//�Ƃ��ƂƎ��̍��W��
					}
					if (smsqtmp > cutoffv) continue;//�Ƃ��ƂƎ��̍��W��
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


		//100���̌v�Z���I�����
		//black
		score[SKLNUM] = Match100_black(iiii, cutoffv_);
		//white
		score[SKLNUM + 1] = Match100_white(iiii, cutoffv_);

		vi idx(SKLNUM2);
		for (int i = 0; i < SKLNUM2; i++)idx[i] = i;
		mysort2(score, idx);
		for (int i = 0; i < SKLNUM2; i++) ptr2[iiii * SKLNUM2 * 2 + i] = score[i];
		for (int i = 0; i < SKLNUM2; i++) ptr2[iiii * SKLNUM2 * 2 + i + SKLNUM2] = idx[i];
	}
}




//�e���v���[�g�摜��o�^ �Fcut���Ĕz��Ɋi�[
//int gid,int xsize,int ysize,array mrefarray
static void SetUserTpBuf(void)
{
	int gid = code_getdi(0);
	int xx = code_getdi(0);
	int yy = code_getdi(0);

	//mref�̃f�[�^�Bbgrbgrbgrbgrbgr�E�E�E�̏��ō������疄�܂��Ă���
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	unsigned char* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (unsigned char*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B

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




//�L���v�`���摜id�ƃe���v���[�g�摜id(buffer2�̂ق�)�����
//�L���v�`���̒��Ƀe���v���[�g�摜�����邩�A�ŏ����a�̍ŏ��l�ƍ��W���o��
//�ŏ����a�̃J�b�g�I�t�l��HSP�������
//int gid,int cutoffv,array outarray[3],int leftx,int rightx
static void MatchManually(void)
{
	int gid = code_getdi(0);
	int cutoffv = code_getdi(0);
	int gxsz = buffer2[gid][0].size();
	int gysz = buffer2[gid].size();

	//�o�͊i�[�f�[�^�Boutsm,outx,outy�E�E�E�̏��Ŗ��߂�
	PVal* pval2;
	APTR aptr2;	//�z��ϐ��̎擾
	aptr2 = code_getva(&pval2);//	���͕ϐ��̌^�Ǝ��̂̃|�C���^���擾
	HspVarProc* phvp2;
	int* ptr2;
	phvp2 = exinfo->HspFunc_getproc(pval2->flag);	//�^����������HspVarProc�\���̂ւ̃|�C���^
	ptr2 = (int*)(phvp2->GetPtr(pval2));					//�f�[�^�ipval1�j�̎��Ԃ�����擪�|�C���^���擾�B

	int leftx = code_getdi(0);//�����J�n��
	int rightx = code_getdi(0);//�����I���E
	leftx = max(leftx, 0);
	rightx = min(SLTSX - gxsz, rightx);


	int outsm = 2000000000;//�ŏI�I�ȍŏ����a
	int outx = -1;//�ŏI�I��x���W
	int outy = -1;//�ŏI�I��y���W
	for (int i = 1; i < SLTSY - gysz - 1; i++)
	{
		for (int j = leftx + 1; j < rightx - 1; j++)
		{
			//i,j�̍��W����݂č��v���邩�B����4�_���������Ĉ�Ԍ덷�����Ȃ��̂��̗p���Ă悢�Ƃ���
			int smsqtmp = 0;//�ŏ����a
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
				if (smsqtmp > cutoffv)break;//�Ƃ��ƂƎ��̍��W��
			}
			if (smsqtmp > cutoffv)continue;//�Ƃ��ƂƎ��̍��W��

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
	//		���s���� (���ߎ��s���ɌĂ΂�܂�)
	//
	code_next();							// ���̃R�[�h���擾(�ŏ��ɕK���K�v�ł�)

	switch( cmd ) {							// �T�u�R�}���h���Ƃ̕���

	case 0x00:								// newcmd
		p1 = code_getdi( 123 );		// �����l���擾(�f�t�H���g123)
		stat = p1;					// �V�X�e���ϐ�stat�ɑ��
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

static int ref_ival;						// �Ԓl�̂��߂̕ϐ�

static void *reffunc( int *type_res, int cmd )
{
	//		�֐��E�V�X�e���ϐ��̎��s���� (�l�̎Q�Ǝ��ɌĂ΂�܂�)
	//
	//			'('�Ŏn�܂邩�𒲂ׂ�
	//
	if ( *type != TYPE_MARK ) puterror( HSPERR_INVALID_FUNCPARAM );
	if ( *val != '(' ) puterror( HSPERR_INVALID_FUNCPARAM );
	code_next();


	switch( cmd ) {							// �T�u�R�}���h���Ƃ̕���

	case 0x00:								// newcmd

		p1 = code_geti();				// �����l���擾(�f�t�H���g�Ȃ�)
		ref_ival = p1 * 2;				// �Ԓl��ival�ɐݒ�
		break;

	default:
		puterror( HSPERR_UNSUPPORTED_FUNCTION );
	}

	//			'('�ŏI��邩�𒲂ׂ�
	//
	if ( *type != TYPE_MARK ) puterror( HSPERR_INVALID_FUNCPARAM );
	if ( *val != ')' ) puterror( HSPERR_INVALID_FUNCPARAM );
	code_next();

	*type_res = HSPVAR_FLAG_INT;			// �Ԓl�̃^�C�v�𐮐��Ɏw�肷��
	return (void *)&ref_ival;
}


/*------------------------------------------------------------*/

static int termfunc( int option )
{
	//		�I������ (�A�v���P�[�V�����I�����ɌĂ΂�܂�)
	//
	return 0;
}

/*------------------------------------------------------------*/

static int eventfunc( int event, int prm1, int prm2, void *prm3 )
{
	//		�C�x���g���� (HSP�C�x���g�������ɌĂ΂�܂�)
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
	//		DLL�G���g���[ (��������K�v�͂���܂���)
	//
	return TRUE;
}


EXPORT void WINAPI hsp3MHRise( HSP3TYPEINFO *info )
{
	//		�v���O�C�������� (���s�E�I��������o�^���܂�)
	//
	hsp3sdk_init( info );			// SDK�̏�����(�ŏ��ɍs�Ȃ��ĉ�����)
	info->cmdfunc = cmdfunc;		// ���s�֐�(cmdfunc)�̓o�^
	info->reffunc = reffunc;		// �Q�Ɗ֐�(reffunc)�̓o�^
	info->termfunc = termfunc;		// �I���֐�(termfunc)�̓o�^

	/*
	//	�C�x���g�R�[���o�b�N�𔭐�������C�x���g��ʂ�ݒ肷��
    info->option = HSPEVENT_ENABLE_GETKEY;
	info->eventfunc = eventfunc;	// �C�x���g�֐�(eventfunc)�̓o�^
	*/
}



/*----------------------------------------------------------------*/
