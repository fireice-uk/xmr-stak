#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

template <typename T>
class thdq
{
  public:
	T pop()
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while(queue_.empty())
		{
			cond_.wait(mlock);
		}
		auto item = std::move(queue_.front());
		queue_.pop();
		return item;
	}

	void pop(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		while(queue_.empty())
		{
			cond_.wait(mlock);
		}
		item = queue_.front();
		queue_.pop();
	}

	void push(const T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		queue_.push(item);
		mlock.unlock();
		cond_.notify_one();
	}

	void push(T&& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		queue_.push(std::move(item));
		mlock.unlock();
		cond_.notify_one();
	}

  private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;
};
