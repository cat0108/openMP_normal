#include<iostream>
#include<Windows.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<omp.h>
using namespace std;
alignas(16) float gdata[10000][10000];//���ж������
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];
int Num_thread = 8;
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j] = gdata2[i][j] = gdata3[i][j];
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
				gdata3[i][j] += gdata3[k][j];
			}
		}
	}

}

void Normal_alg(int N)
{
	int i, j, k;
	for (k = 0; k < N; k++)
	{
		for (j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (i = k + 1; i < N; i++)
		{
			for (j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}


void omp_SSE(int n)
{
#pragma omp parallel num_threads(Num_thread)
	{
		int i, j, k;
		__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
		for (k = 0; k < n; k++)
		{
#pragma omp single
			{
				float temp[4] = { gdata3[k][k],gdata3[k][k],gdata3[k][k],gdata3[k][k] };
				r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
				for (j = k + 1; j + 4 <= n; j += 4)
				{
					r1 = _mm_loadu_ps(gdata3[k] + j);
					r1 = _mm_div_ps(r1, r0);//������������λ���
					_mm_storeu_ps(gdata3[k], r1);//���������·Ż��ڴ�
				}
				//��ʣ�಻��4�������ݽ�����Ԫ
				for (j; j < n; j++)
				{
					gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
				}
				gdata3[k][k] = 1.0;
				//���϶�Ӧ������һ������ѭ���Ż���SIMD
			}
#pragma omp for
			for (i = k + 1; i < n; i++)
			{
				float temp2[4] = { gdata3[i][k],gdata3[i][k],gdata3[i][k],gdata3[i][k] };
				r0 = _mm_loadu_ps(temp2);
				for (j = k + 1; j + 4 <= n; j += 4)
				{
					r1 = _mm_loadu_ps(gdata3[k] + j);
					r2 = _mm_loadu_ps(gdata3[i] + j);
					r3 = _mm_mul_ps(r0, r1);
					r2 = _mm_sub_ps(r2, r3);
					_mm_storeu_ps(gdata3[i] + j, r2);
				}
				for (j; j < n; j++)
				{
					gdata3[i][j] = gdata3[i][j] - (gdata3[i][k] * gdata3[k][j]);
				}
				gdata3[i][k] = 0;
			}
		}
	}
}



//��ȫ�������Ż�
void Par_alg_all(int n)
{
	int i, j, k;
	__m128 r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		float temp[4] = { gdata[k][k],gdata[k][k],gdata[k][k],gdata[k][k] };
		r0 = _mm_loadu_ps(temp);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = _mm_loadu_ps(gdata[k] + j);
			r1 = _mm_div_ps(r1, r0);//������������λ���
			_mm_storeu_ps(gdata[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata[k][j] = gdata[k][j] / gdata[k][k];
		}
		gdata[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{
			float temp2[4] = { gdata[i][k],gdata[i][k],gdata[i][k],gdata[i][k] };
			r0 = _mm_loadu_ps(temp2);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = _mm_loadu_ps(gdata[k] + j);
				r2 = _mm_loadu_ps(gdata[i] + j);
				r3 = _mm_mul_ps(r0, r1);
				r2 = _mm_sub_ps(r2, r3);
				_mm_storeu_ps(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
	}
}




int main()
{
	LARGE_INTEGER fre, begin, end;
	double gettime;
	int n;
	cin >> n;
	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Initialize(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "intial time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Normal_alg(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "normal time: " << gettime << " ms" << endl;

	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	omp_SSE(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "omp time: " << gettime << " ms" << endl;


	QueryPerformanceFrequency(&fre);
	QueryPerformanceCounter(&begin);
	Par_alg_all(n);
	QueryPerformanceCounter(&end);
	gettime = (double)((end.QuadPart - begin.QuadPart) * 1000.0) / (double)fre.QuadPart;
	cout << "SSE time: " << gettime << " ms" << endl;
}
